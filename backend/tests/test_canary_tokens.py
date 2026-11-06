"""Unit tests for the canary-token module (stdlib unittest only, so
they run anywhere: `python -m unittest test_canary_tokens -v`)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import canary_tokens as ct  # noqa: E402


class TestGenerators(unittest.TestCase):
    def test_aws_keypair_shape(self):
        pair = ct.generate_aws_keypair()
        self.assertEqual(ct.identify_family(pair["access_key_id"]), "aws_access_key_id")
        self.assertEqual(ct.identify_family(pair["secret_access_key"]), "aws_secret_access_key")
        self.assertEqual(len(pair["secret_access_key"]), 40)

    def test_github_pat_shape(self):
        token = ct.generate_github_pat()
        self.assertTrue(token["token"].startswith("ghp_"))
        self.assertEqual(ct.identify_family(token["token"]), "github_pat")

    def test_api_key_shape_and_prefix(self):
        token = ct.generate_api_key()
        self.assertEqual(ct.identify_family(token["token"]), "api_key")
        custom = ct.generate_api_key(prefix="acme")
        self.assertTrue(custom["token"].startswith("acme_"))

    def test_api_key_rejects_bad_prefix(self):
        with self.assertRaises(ValueError):
            ct.generate_api_key(prefix="Bad Prefix!")

    def test_postgres_dsn_shape(self):
        token = ct.generate_postgres_dsn()
        self.assertEqual(ct.identify_family(token["dsn"]), "postgres_dsn")
        self.assertIn(token["username"], token["dsn"])

    def test_generated_values_are_unique(self):
        values = set()
        for _ in range(200):
            values.update(ct.token_values(ct.generate_aws_keypair()))
            values.update(ct.token_values(ct.generate_github_pat()))
            values.update(ct.token_values(ct.generate_api_key()))
        self.assertEqual(len(values), 200 * 4)  # no collisions, no repeats

    def test_identify_family_rejects_plain_text(self):
        self.assertIsNone(ct.identify_family("just some ordinary text"))
        self.assertIsNone(ct.identify_family("AKIA"))  # too short
        self.assertIsNone(ct.identify_family(""))


class TestRegistry(unittest.TestCase):
    def test_register_and_check_roundtrip(self):
        registry = ct.CanaryRegistry()
        token = registry.generate_and_register(
            "aws", planted_in="repo:legacy-infra/config.py", actor="manoj"
        )
        hit = registry.check(token["access_key_id"])
        self.assertIsNotNone(hit)
        self.assertEqual(hit["planted_in"], "repo:legacy-infra/config.py")
        self.assertEqual(hit["family"], "aws")
        # The secret half resolves to the same plant.
        self.assertIs(registry.check(token["secret_access_key"]), hit)

    def test_check_miss_returns_none(self):
        registry = ct.CanaryRegistry()
        registry.generate_and_register("github_pat", planted_in="gist:notes")
        other = ct.generate_github_pat()
        self.assertIsNone(registry.check(other["token"]))
        self.assertIsNone(registry.check("not a token at all"))

    def test_registry_never_stores_raw_secrets(self):
        registry = ct.CanaryRegistry()
        token = registry.generate_and_register("postgres_dsn", planted_in="file:.env.example")
        stored = repr(registry.get_plants()) + repr(registry.__dict__)
        for raw in ct.token_values(token):
            self.assertNotIn(raw, stored)

    def test_unknown_family_rejected(self):
        registry = ct.CanaryRegistry()
        with self.assertRaises(ValueError):
            registry.generate_and_register("credit_card", planted_in="nowhere")


if __name__ == "__main__":
    unittest.main()
