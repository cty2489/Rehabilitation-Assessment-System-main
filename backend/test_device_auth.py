import unittest

from device_auth import (
    DeviceTokenConfigError,
    authenticate_device_token,
    credential_count,
    parse_named_tokens,
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


if __name__ == "__main__":
    unittest.main()
