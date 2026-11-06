"""API tests for the canary-token endpoints (POST /api/canary/generate,
POST /api/canary/check) plus registry persistence.

Run from the repo root:  python -m unittest backend.tests.test_canary_api -v
Uses FastAPI's TestClient against a throwaway registry file.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

_TMP = tempfile.mkdtemp(prefix="canary-api-test-")
os.environ["CANARY_REGISTRY_PATH"] = str(Path(_TMP) / "canary_registry.json")

from fastapi.testclient import TestClient  # noqa: E402

from backend import main as main_module  # noqa: E402
from backend.canary_tokens import CanaryRegistry  # noqa: E402


class TestCanaryApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main_module.app)
        resp = cls.client.post("/auth/signup", json={
            "username": "canary.tester",
            "password": "CanaryTest@123",
            "full_name": "Canary Tester",
        })
        assert resp.status_code == 200, resp.text
        resp = cls.client.post("/auth/token", data={
            "username": "canary.tester", "password": "CanaryTest@123",
        })
        assert resp.status_code == 200, resp.text
        cls.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    def test_generate_aws_registers_fingerprint_only(self):
        resp = self.client.post("/api/canary/generate", headers=self.headers, json={
            "family": "aws", "planted_in": "github.com/acme/infra/.env",
        })
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        token, plant = body["token"], body["plant"]
        self.assertTrue(token["access_key_id"].startswith("AKIA"))
        self.assertEqual(plant["planted_in"], "github.com/acme/infra/.env")
        self.assertEqual(plant["family"], "aws")
        # The plant record must never contain the raw secret values.
        plant_json = json.dumps(plant)
        self.assertNotIn(token["access_key_id"], plant_json)
        self.assertNotIn(token["secret_access_key"], plant_json)
        self.assertEqual(len(plant["fingerprints"]), 2)

    def test_check_hit_returns_plant(self):
        gen = self.client.post("/api/canary/generate", headers=self.headers, json={
            "family": "github_pat", "planted_in": "leaked dump, paste site",
        }).json()
        value = gen["token"]["token"]
        resp = self.client.post("/api/canary/check", headers=self.headers,
                                json={"value": value})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["hit"])
        self.assertEqual(body["plant"]["planted_in"], "leaked dump, paste site")
        self.assertEqual(body["identified_family"], "github_pat")

    def test_check_miss_for_unregistered_value(self):
        resp = self.client.post("/api/canary/check", headers=self.headers,
                                json={"value": "ghp_" + "A" * 36})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["hit"])
        self.assertIsNone(body["plant"])
        self.assertEqual(body["identified_family"], "github_pat")

    def test_generate_unknown_family_is_400(self):
        resp = self.client.post("/api/canary/generate", headers=self.headers, json={
            "family": "carrier_pigeon", "planted_in": "roof",
        })
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_endpoints_require_auth(self):
        resp = self.client.post("/api/canary/generate",
                                json={"family": "aws", "planted_in": "x"})
        self.assertEqual(resp.status_code, 401)
        resp = self.client.post("/api/canary/check", json={"value": "x"})
        self.assertEqual(resp.status_code, 401)

    def test_registry_survives_reload_and_stays_fingerprint_only(self):
        gen = self.client.post("/api/canary/generate", headers=self.headers, json={
            "family": "postgres_dsn", "planted_in": "docker-compose.yml (staging)",
        }).json()
        dsn = gen["token"]["dsn"]
        # A brand-new registry over the same file must still know the token.
        reloaded = CanaryRegistry(storage_path=os.environ["CANARY_REGISTRY_PATH"])
        record = reloaded.check(dsn)
        self.assertIsNotNone(record)
        self.assertEqual(record["planted_in"], "docker-compose.yml (staging)")
        # And the raw DSN (with its embedded password) is nowhere on disk.
        on_disk = Path(os.environ["CANARY_REGISTRY_PATH"]).read_text(encoding="utf-8")
        self.assertNotIn(dsn, on_disk)


if __name__ == "__main__":
    unittest.main()
