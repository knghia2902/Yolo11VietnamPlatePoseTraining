import base64
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from PIL import Image

TOOLS = Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'
sys.path.insert(0, str(TOOLS))
from pose_model_test_server import ModelTester, ModelTestHandler
from register_test_model import register


class ModelTestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        folder = self.root/'models'/'test-run';folder.mkdir(parents=True)
        (folder/'best.pt').write_bytes(b'synthetic weights')
        (folder/'model.json').write_text(json.dumps(dict(name='Test model', sha256=hashlib.sha256(b'synthetic weights').hexdigest())))
        self.tester = ModelTester(self.root/'models', sys.executable, self.root)
        image = io.BytesIO();Image.new('RGB', (32,32)).save(image, format='PNG')
        self.request = dict(model='test-run', image=base64.b64encode(image.getvalue()).decode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_registry_and_path_traversal(self):
        self.assertEqual(self.tester.models()[0]['id'], 'test-run')
        with self.assertRaises(ValueError):self.tester.predict(dict(self.request,model='../test-run'))
        with self.assertRaises(ValueError):self.tester.predict(dict(self.request,model='missing'))

    def test_changed_weights_rejected(self):
        (self.root/'models/test-run/best.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checksum'):self.tester.predict(self.request)

    def test_register_does_not_overwrite_previous_version(self):
        source=self.root/'trusted.pt';source.write_bytes(b'test weights')
        folder=register(source,self.root/'registered','version-1','Synthetic model',1)
        before=(folder/'best.pt').read_bytes()
        with self.assertRaises(FileExistsError):register(source,self.root/'registered','version-1','Changed',2)
        self.assertEqual(before,(folder/'best.pt').read_bytes())
        with self.assertRaises(ValueError):register(source,self.root/'registered','../invalid','Invalid',1)

    def test_invalid_image_and_busy_rejected(self):
        with self.assertRaises(ValueError):self.tester.predict(dict(self.request,image='not base64'))
        self.tester.lock.acquire()
        try:
            with self.assertRaises(RuntimeError):self.tester.predict(self.request)
        finally:self.tester.lock.release()

    def test_inference_is_transient_and_does_not_change_weights(self):
        before=(self.root/'models/test-run/best.pt').read_bytes()
        temporary=[]
        def run(args, **kwargs):
            temporary.append(Path(args[-1]).parent)
            Path(args[-1]).write_text(json.dumps(dict(detections=[], elapsed_seconds=0.1)))
            return type('Result',(),dict(returncode=0))()
        with patch('pose_model_test_server.subprocess.run',side_effect=run):
            result=self.tester.predict(self.request)
        self.assertEqual(result['detections'],[])
        self.assertEqual(before,(self.root/'models/test-run/best.pt').read_bytes())
        self.assertFalse(temporary[0].exists())
        self.assertFalse((self.root/'curation.sqlite').exists())

    def test_authenticated_http_and_cross_origin_protection(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),ModelTestHandler)
        server.tester=self.tester
        server.networks=[ipaddress.ip_network('127.0.0.0/8')]
        server.expected_auth='Basic '+base64.b64encode(b'test:synthetic').decode()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        def request(path, body=None, origin=None, authorized=True):
            headers={'Content-Type':'application/json','X-Review-Request':'1'}
            if authorized:headers['Authorization']=server.expected_auth
            if origin:headers['Origin']=origin
            return urllib.request.urlopen(urllib.request.Request(url+path,data=body,headers=headers))
        try:
            with self.assertRaises(urllib.error.HTTPError) as denied:request('/api/model-test/models',authorized=False)
            self.assertEqual(denied.exception.code,401)
            with request('/api/model-test/models') as r:self.assertEqual(len(json.load(r)['models']),1)
            with request('/model-test') as r:
                page=r.read().decode()
                self.assertIn('Test model YOLO point',page)
                self.assertIn('createImageBitmap',page)
                self.assertNotIn('data:image',page)
                self.assertNotIn('new Image()',page)
                self.assertIn('img-src \'self\'',r.headers['Content-Security-Policy'])
                self.assertIn('--bg-app:#dee3ed',page.lower())
                self.assertIn('height:28px',page)
            with self.assertRaises(urllib.error.HTTPError) as cross:request('/api/model-test/predict',b'{}','http://other.example')
            self.assertEqual(cross.exception.code,403)
            with self.assertRaises(urllib.error.HTTPError) as invalid:request('/api/model-test/predict',b'[]',url)
            self.assertEqual(invalid.exception.code,400)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
