"""Private LAN server tests with synthetic fixtures only."""
import base64
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest

import test_pose_dataset as fixtures

sys.path.insert(0, str(fixtures.TOOL.parent))
import serve_pose_review as server
sys.path.pop(0)


class ServerTests(unittest.TestCase):
    scan = fixtures.DatasetTests.scan

    def setUp(self):
        fixtures.DatasetTests.setUp(self)
        self.manifest = self.scan()
        self.httpd = server.ReviewServer(("127.0.0.1", 0), self.root / "workspace",
                                         {"username": "test", "password": "synthetic-only"}, ["127.0.0.0/8"])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(self, method="GET", path="/", payload=None, auth=True, origin=None):
        headers = {}
        if auth:
            headers["Authorization"] = "Basic " + base64.b64encode(b"test:synthetic-only").decode()
        if method == "POST":
            headers.update({"Content-Type": "application/json", "X-Review-Request": "1"})
        if origin:
            headers["Origin"] = origin
        connection = http.client.HTTPConnection(*self.httpd.server_address, timeout=5)
        connection.request(method, path, json.dumps(payload) if payload is not None else None, headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def update(self, **extra):
        row = self.manifest["records"][0]
        edit = {key: row.get(key, "" if key == "reviewer" else None) for key in server.EDIT_FIELDS}
        edit.update(reviewed=True, group_reviewed=True, reviewer="tester")
        return dict(id=row["id"], version=0, cursor=1, edit=edit, **extra)

    def test_http_auth_and_private_files(self):
        self.assertEqual(self.request(auth=False)[0], 401)
        self.assertEqual(self.request()[0], 200)
        for path in ("/manifest.json", "/saved-reviews/", "/../manifest.json", "/api/export"):
            self.assertEqual(self.request(path=path, auth=False)[0], 401)
        self.assertEqual(self.request(path="/../manifest.json")[0], 404)
        self.assertEqual(self.request(path="/" + self.manifest["records"][0]["image"])[0], 200)

    def test_persist_reload_export_and_conflict(self):
        update = self.update()
        status, data = self.request("POST", "/api/record", update)
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)["version"], 1)
        self.assertEqual(self.request("POST", "/api/record", update)[0], 409)
        reopened = server.ReviewStore(self.root / "workspace")
        snapshot, cursor = reopened.snapshot()
        self.assertTrue(snapshot["records"][0]["reviewed"])
        self.assertEqual(cursor, 1)
        status, data = self.request(path="/api/export")
        self.assertTrue(json.loads(data)["records"][0]["reviewed"])
        update["version"] = 1
        update["edit"]["reviewer"] = "second tester"
        self.assertEqual(self.request("POST", "/api/record", update)[0], 200)
        self.assertEqual(len(list((self.root / "workspace/saved-reviews/history").rglob("*.json"))), 1)

    def test_invalid_approval_and_cross_origin_rejected(self):
        update = self.update()
        self.assertEqual(self.request("POST", "/api/record", update, origin="http://example.invalid")[0], 403)
        update["edit"]["points"] = []
        self.assertEqual(self.request("POST", "/api/record", update)[0], 400)
        update["edit"]["reviewed"] = False
        self.assertEqual(self.request("POST", "/api/record", update)[0], 200)

    def test_source_fields_cannot_be_changed(self):
        update = self.update()
        update["edit"]["image"] = "../secret"
        self.assertEqual(self.request("POST", "/api/record", update)[0], 400)

    def test_second_server_cannot_write_same_workspace(self):
        with self.assertRaises(OSError):
            server.ReviewServer(("127.0.0.1", 0), self.root / "workspace",
                               {"username": "test", "password": "synthetic-only"}, ["127.0.0.0/8"])

    def test_exclusion_moves_image_hides_http_and_can_restore(self):
        row = self.manifest["records"][0]
        active = self.root / "workspace" / row["image"]
        archived = self.root / "workspace/excluded-images" / active.name
        update = self.update()
        update["edit"]["exclude"] = True
        self.assertEqual(self.request("POST", "/api/record", update)[0], 200)
        self.assertFalse(active.exists())
        self.assertTrue(archived.exists())
        self.assertEqual(self.request(path="/" + row["image"])[0], 404)
        reopened = server.ReviewStore(self.root / "workspace")
        self.assertTrue(reopened.snapshot()[0]["records"][0]["exclude"])
        update["version"] = 1
        update["edit"]["exclude"] = False
        self.assertEqual(self.request("POST", "/api/record", update)[0], 200)
        self.assertTrue(active.exists())
        self.assertFalse(archived.exists())

    def test_exclusion_recovers_move_before_uncommitted_save(self):
        row = self.manifest["records"][0]
        active = self.root / "workspace" / row["image"]
        archived = self.root / "workspace/excluded-images" / active.name
        active.rename(archived)
        server.ReviewStore(self.root / "workspace")
        self.assertTrue(active.exists())
        self.assertFalse(archived.exists())


if __name__ == "__main__":
    unittest.main()
