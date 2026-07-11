import unittest

from device_auth import (
    DeviceTokenConfigError,
    authenticate_device_token,
    credential_count,
    generate_device_token,
    parse_named_tokens,
    token_digest,
    token_hint,
)


class DeviceAuthTests(unittest.TestCase):
    def test_legacy_and_named_tokens_are_supported(self) -> None:
        raw = '{"device_002":"token-two","device_003":"token-three"}'
        self.assertEqual(credential_count("legacy-token", raw), 3)
        self.assertTrue(authenticate_device_token("legacy-token", "legacy-token", raw).legacy)
        self.assertEqual(
            authenticate_device_token("token-two", "legacy-token", raw).device_id,
            "device_002",
        )
        self.assertIsNone(authenticate_device_token("wrong", "legacy-token", raw))

    def test_invalid_or_duplicate_config_is_rejected(self) -> None:
        with self.assertRaises(DeviceTokenConfigError):
            parse_named_tokens("not-json")
        with self.assertRaises(DeviceTokenConfigError):
            parse_named_tokens('{"a":"same","b":"same"}')
        with self.assertRaises(DeviceTokenConfigError):
            credential_count("same", '{"a":"same"}')

    def test_generated_token_has_stable_digest_and_masked_hint(self) -> None:
        token = generate_device_token()
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(len(token_digest(token)), 64)
        self.assertNotIn(token, token_hint(token))


if __name__ == "__main__":
    unittest.main()
