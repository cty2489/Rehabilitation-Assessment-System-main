import unittest

from admin_auth import browser_origin_allowed, issue_session_token, verify_session_token
from schemas import AuthLoginResponse


class AdminAuthTests(unittest.TestCase):
    def test_signed_session_expires_and_rejects_tampering(self):
        token = issue_session_token("admin", "secret-key", 60, now=1000)
        self.assertTrue(verify_session_token(token, "admin", "secret-key", now=1059))
        self.assertFalse(verify_session_token(token, "admin", "secret-key", now=1060))
        self.assertFalse(verify_session_token(token, "admin", "secret-key", now=1061))
        self.assertFalse(verify_session_token(token + "x", "admin", "secret-key", now=1050))
        self.assertFalse(verify_session_token(token, "other", "secret-key", now=1050))

    def test_malformed_base64_is_rejected_without_raising(self):
        self.assertFalse(verify_session_token("@@@.@@@", "admin", "secret-key", now=1000))
        self.assertFalse(verify_session_token("not-a-token", "admin", "secret-key", now=1000))

    def test_cookie_write_origin_must_match_public_origin(self):
        self.assertTrue(
            browser_origin_allowed(
                "https://demo.example.com:8443",
                "",
                "https://demo.example.com:8443",
            )
        )
        self.assertTrue(
            browser_origin_allowed(
                "",
                "https://demo.example.com:8443/settings",
                "https://demo.example.com",
            )
        )
        self.assertFalse(
            browser_origin_allowed(
                "https://attacker.example.com",
                "",
                "https://demo.example.com:8443",
            )
        )
        self.assertFalse(browser_origin_allowed("", "", "https://demo.example.com:8443"))

    def test_login_response_never_exposes_session_token(self):
        payload = AuthLoginResponse(user="admin", expires_in=3600).model_dump()
        self.assertEqual(payload, {"user": "admin", "expires_in": 3600})
        self.assertNotIn("access_token", payload)


if __name__ == "__main__":
    unittest.main()
