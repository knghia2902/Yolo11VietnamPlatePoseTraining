"""Authenticated, read-only model testing alongside the annotation review server."""
import argparse
import base64
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import threading
from urllib.parse import urlsplit
from http.server import ThreadingHTTPServer
from PIL import Image, ImageOps
from curate_acs_training_pool import ExpansionReviewHandler, ExpansionReviewStore


class ModelTester:
    def __init__(self, model_root, python, workspace):
        self.root = Path(model_root).resolve()
        self.python = str(Path(python).resolve())
        self.workspace = Path(workspace).resolve()
        self.lock = threading.Lock()

    def models(self):
        result = []
        for file in sorted(self.root.glob('*/model.json'), reverse=True):
            data = json.loads(file.read_text(encoding='utf-8'))
            weights = file.parent / 'best.pt'
            if weights.is_file() and not weights.is_symlink():
                result.append(dict(id=file.parent.name, name=data['name'], trained_at=data.get('trained_at', ''),
                                   best_epoch=data.get('best_epoch'), sha256=data['sha256']))
        return result

    def predict(self, request):
        model_id = request.get('model', '')
        if not isinstance(model_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', model_id):
            raise ValueError('Model không hợp lệ')
        entry = next((m for m in self.models() if m['id'] == model_id), None)
        if entry is None:
            raise ValueError('Không tìm thấy model')
        weights = (self.root / model_id / 'best.pt').resolve()
        if not weights.is_relative_to(self.root):
            raise ValueError('Model nằm ngoài thư mục cho phép')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Checksum model không khớp')
        if request.get('id'):
            image_id = request['id']
            if not isinstance(image_id, str) or not re.fullmatch('[a-f0-9]{64}', image_id):
                raise ValueError('Ảnh không hợp lệ')
            with sqlite3.connect(f'{(self.workspace / "curation.sqlite").as_uri()}?mode=ro', uri=True) as db:
                row = db.execute('SELECT path FROM images WHERE id=?', (image_id,)).fetchone()
            if not row:
                raise ValueError('Không tìm thấy ảnh')
            data = Path(row[0]).read_bytes()
        else:
            try:
                data = base64.b64decode(request.get('image', ''), validate=True)
            except (ValueError, TypeError):
                raise ValueError('Ảnh tải lên không hợp lệ') from None
        if not 0 < len(data) <= 15 * 1024 * 1024:
            raise ValueError('Ảnh tối đa 15 MB')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 25_000_000:
                    raise ValueError('Ảnh tối đa 25 triệu pixel')
                image = ImageOps.exif_transpose(image).convert('RGB')
                normalized = io.BytesIO()
                image.save(normalized, format='JPEG', quality=95)
        except (OSError, Image.DecompressionBombError):
            raise ValueError('Không đọc được ảnh') from None
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Đang test ảnh khác, vui lòng thử lại sau')
        try:
            with tempfile.TemporaryDirectory(prefix='anpr-model-test-') as folder:
                source = Path(folder) / 'input.jpg'
                output = Path(folder) / 'result.json'
                source.write_bytes(normalized.getvalue())
                completed = subprocess.run([self.python, '-X', 'utf8', str(Path(__file__).with_name('pose_model_test_worker.py')),
                    str(weights), str(source), str(output)], capture_output=True, timeout=90,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if completed.returncode or not output.is_file():
                    raise RuntimeError('Không chạy được model; kiểm tra môi trường inference trên máy chủ')
                result = json.loads(output.read_text(encoding='utf-8'))
                result['image'] = base64.b64encode(normalized.getvalue()).decode('ascii')
                result['model'] = entry
                return result
        except subprocess.TimeoutExpired:
            raise RuntimeError('Model chạy quá 90 giây, vui lòng thử ảnh nhỏ hơn') from None
        finally:
            self.lock.release()


class ModelTestHandler(ExpansionReviewHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path not in ('/model-test', '/api/model-test/models'):
            return super().do_GET()
        if not self.authenticated():
            return
        if path == '/model-test':
            return self.respond(200, Path(__file__).with_name('review_model_test.html').read_bytes(), 'text/html; charset=utf-8')
        self.respond(200, dict(models=self.server.tester.models()))

    def do_POST(self):
        if self.path != '/api/model-test/predict':
            return super().do_POST()
        if not self.authenticated():
            return
        if (self.headers.get('Origin') != 'http://' + self.headers.get('Host', '') or
            self.headers.get('X-Review-Request') != '1' or self.headers.get('Content-Type') != 'application/json'):
            return self.respond(403, dict(error='Invalid origin or content type'))
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 21 * 1024 * 1024:
                raise ValueError('Ảnh tải lên quá lớn')
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError('Yêu cầu không hợp lệ')
            self.respond(200, self.server.tester.predict(request))
        except (ValueError, TypeError, KeyError) as exc:
            self.respond(400, dict(error=str(exc)))
        except RuntimeError as exc:
            self.respond(503, dict(error=str(exc)))
        except OSError:
            self.respond(400, dict(error='Không đọc được tệp ảnh hoặc model'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True)
    p.add_argument('--credentials', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--inference-python', required=True)
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--allow-network', action='append', default=['127.0.0.0/8'])
    a = p.parse_args()
    server = ThreadingHTTPServer((a.bind, a.port), ModelTestHandler)
    server.daemon_threads = True
    server.store = ExpansionReviewStore(a.workspace)
    server.tester = ModelTester(a.model_root, a.inference_python, a.workspace)
    server.networks = [ipaddress.ip_network(n) for n in a.allow_network]
    login = json.loads(Path(a.credentials).read_text(encoding='utf-8'))
    server.expected_auth = 'Basic ' + base64.b64encode((login['username'] + ':' + login['password']).encode()).decode()
    server.template = Path(__file__).with_name('review_expansion.html').read_text(encoding='utf-8')
    print('ANPR review and model testing ready', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
