"""Private LAN review server. Images and annotation state remain on this PC."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from prepare_pose_dataset import ORDER, valid_points

EDIT_FIELDS = ("points", "reviewed", "group", "group_reviewed", "reviewer", "exclude")


class ConflictError(ValueError):
    pass


class ReviewStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.manifest = json.loads((self.workspace / "manifest.json").read_text(encoding="utf-8"))
        self.originals = {r["id"]: r for r in self.manifest["records"]}
        self.images = {"/" + r["image"]: self.workspace / r["image"] for r in self.originals.values()}
        self.state_dir = self.workspace / "saved-reviews"
        self.state_dir.mkdir(exist_ok=True)
        self.excluded_dir = self.workspace / "excluded-images"
        self.excluded_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.updates = {}
        for file in self.state_dir.glob("*.json"):
            state = json.loads(file.read_text(encoding="utf-8"))
            if state["source_sha256"] != self.manifest["source_sha256"] or file.stem not in self.originals:
                raise ValueError("Saved reviews belong to another dataset")
            self.updates[file.stem] = state
        # Recover the image location if a process stopped between move and save.
        quarantined = {file.name for file in self.excluded_dir.iterdir() if file.is_file()}
        for image_id, original in self.originals.items():
            if image_id in self.updates or Path(original["image"]).name in quarantined:
                self.place_image(image_id, self.updates.get(image_id, {}).get("edit", {}).get("exclude", False))

    def place_image(self, image_id, excluded):
        source = self.images["/" + self.originals[image_id]["image"]]
        target = self.excluded_dir / source.name
        # All moves must stay within this explicit review workspace.
        if (self.workspace not in source.resolve().parents or self.workspace not in target.resolve().parents or
            (self.workspace / "images").resolve() not in source.resolve().parents or self.excluded_dir.resolve() not in target.resolve().parents):
            raise ValueError("Image move is outside the review workspace")
        if source.is_symlink() or target.is_symlink():
            raise ValueError("Image symlinks are not supported")
        origin, destination = (source, target) if excluded else (target, source)
        if origin.exists():
            if destination.exists():
                raise ValueError("Both active and excluded image copies exist; reconcile before moving")
            os.replace(origin, destination)

    def read_image(self, path):
        with self.lock:
            if path not in self.images:
                return None
            image_id = Path(self.images[path]).stem
            if self.updates.get(image_id, {}).get("edit", {}).get("exclude", False):
                return None
            return self.images[path].read_bytes()

    def snapshot(self):
        with self.lock:
            cursor = max(self.updates.values(), key=lambda value: value["saved_at"], default={"cursor": 0})["cursor"]
            records = []
            for original in self.manifest["records"]:
                update = self.updates.get(original["id"], {"version": 0, "edit": {}})
                records.append({**original, **update["edit"], "_version": update["version"]})
        return {**self.manifest, "records": records}, cursor

    def save(self, request):
        image_id = request.get("id")
        if image_id not in self.originals:
            raise ValueError("Unknown image")
        version = request.get("version")
        if type(version) is not int or version < 0:
            raise ValueError("Invalid revision")
        payload = request.get("edit")
        if not isinstance(payload, dict) or set(payload) != set(EDIT_FIELDS):
            raise ValueError("Invalid annotation fields")
        for key in ("reviewed", "group_reviewed", "exclude"):
            if type(payload[key]) is not bool:
                raise ValueError("Invalid boolean")
        for key in ("group", "reviewer"):
            if not isinstance(payload[key], str) or len(payload[key]) > 200:
                raise ValueError("Invalid group or reviewer")
        points = payload["points"]
        if not isinstance(points, list) or len(points) > 4:
            raise ValueError("Invalid corners")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Invalid point")
            if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in point):
                raise ValueError("Invalid coordinate")
        if payload["reviewed"]:
            valid_points(points)
            if not payload["group_reviewed"] or not payload["group"].strip() or not payload["reviewer"].strip():
                raise ValueError("Approval requires reviewer and verified group")
        cursor = request.get("cursor")
        if type(cursor) is not int or not 0 <= cursor < len(self.originals):
            raise ValueError("Invalid cursor")
        with self.lock:
            old = self.updates.get(image_id)
            if (old["version"] if old else 0) != version:
                raise ConflictError("Image was changed on another tab or PC; export local edits before reloading")
            state = dict(source_sha256=self.manifest["source_sha256"], version=version + 1,
                         edit=payload, cursor=cursor, saved_at=time.time_ns())
            if old:
                history = self.state_dir / "history" / image_id
                history.mkdir(parents=True, exist_ok=True)
                previous = history / f"{version}.json"
                if not previous.exists():
                    previous.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            temporary = self.state_dir / (image_id + ".tmp")
            self.place_image(image_id, payload["exclude"])
            try:
                with temporary.open("w", encoding="utf-8") as file:
                    json.dump(state, file, ensure_ascii=False)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.state_dir / (image_id + ".json"))
            except Exception:
                self.place_image(image_id, old["edit"]["exclude"] if old else False)
                raise
            self.updates[image_id] = state
        return {"version": version + 1}


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, workspace, credentials, networks):
        self.state_lock = (Path(workspace) / "review-server.lock").open("a+b")
        try:
            self.state_lock.seek(0, 2)
            if self.state_lock.tell() == 0:
                self.state_lock.write(b"0")
                self.state_lock.flush()
            self.state_lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.state_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.state_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = ReviewStore(workspace)
        except Exception:
            self.state_lock.close()
            raise
        self.networks = [ipaddress.ip_network(n) for n in networks]
        self.expected_auth = "Basic " + base64.b64encode(
            (credentials["username"] + ":" + credentials["password"]).encode()).decode()
        self.template = Path(__file__).with_name("review_pose.html").read_text(encoding="utf-8")
        try:
            super().__init__(address, ReviewHandler)
        except Exception:
            self.state_lock.close()
            raise

    def server_close(self):
        super().server_close()
        self.state_lock.close()


class ReviewHandler(BaseHTTPRequestHandler):
    # Do not log URLs, credentials, image IDs or customer metadata.
    def log_message(self, *_args):
        pass

    def respond(self, status, data, content_type="application/json; charset=utf-8", headers=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def authenticated(self):
        peer = ipaddress.ip_address(self.client_address[0])
        if not any(peer in network for network in self.server.networks):
            self.respond(403, {"error": "Client is outside configured LAN"})
            return False
        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("http://" + host).hostname
            address = ipaddress.ip_address(hostname)
            if not any(address in network for network in self.server.networks):
                raise ValueError("Host outside LAN")
        except ValueError:
            self.respond(403, {"error": "Use the server LAN IP address"})
            return False
        if not hmac.compare_digest(self.headers.get("Authorization", ""), self.server.expected_auth):
            self.respond(401, {"error": "Login required"}, headers={"WWW-Authenticate": 'Basic realm="ANPR review LAN", charset="UTF-8"'})
            return False
        return True

    def do_GET(self):
        if not self.authenticated():
            return
        path = urlsplit(self.path).path
        if path in ("/", "/review.html"):
            manifest, cursor = self.server.store.snapshot()
            payload = json.dumps(manifest, ensure_ascii=True).replace("<", "\\u003c")
            page = self.server.template.replace("/*MANIFEST*/null", payload).replace("/*SERVER*/null", json.dumps({"cursor": cursor}))
            self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/export":
            manifest, _ = self.server.store.snapshot()
            data = {"source_sha256": manifest["source_sha256"], "records": manifest["records"]}
            self.respond(200, data, headers={"Content-Disposition": 'attachment; filename="reviews.json"'})
        elif path in self.server.store.images:
            file = self.server.store.images[path]
            data = self.server.store.read_image(path)
            if data is None:
                self.respond(404, {"error": "Image excluded"})
            else:
                self.respond(200, data, "image/png" if file.suffix == ".png" else "image/jpeg")
        else:
            self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authenticated():
            return
        if self.path != "/api/record":
            self.respond(404, {"error": "Not found"})
            return
        # JSON + a custom header disallow cross-site form writes; verify Origin too.
        origin = self.headers.get("Origin")
        if (self.headers.get("X-Review-Request") != "1" or
            self.headers.get("Content-Type", "").split(";")[0] != "application/json" or
            (origin and origin != "http://" + self.headers.get("Host", ""))):
            self.respond(403, {"error": "Invalid request origin or content type"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("Invalid request")
            result = self.server.store.save(request)
            self.respond(200, result)
        except ConflictError as exc:
            self.respond(409, {"error": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            self.respond(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--credentials", required=True, help="Private JSON file outside checkout")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-network", action="append", default=["127.0.0.0/8"])
    args = parser.parse_args()
    credential_file = Path(args.credentials).resolve()
    repo = Path(__file__).resolve().parents[3]
    if repo == credential_file or repo in credential_file.parents:
        raise ValueError("Credentials must stay outside the repository")
    if not credential_file.exists():
        credential_file.parent.mkdir(parents=True, exist_ok=True)
        with credential_file.open("x", encoding="utf-8") as file:
            json.dump({"username": "reviewer", "password": secrets.token_urlsafe(24)}, file)
    credentials = json.loads(credential_file.read_text(encoding="utf-8"))
    server = ReviewServer((args.bind, args.port), args.workspace, credentials, args.allow_network)
    print(f"ANPR review listening on {args.bind}:{args.port}; credentials: {credential_file}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
