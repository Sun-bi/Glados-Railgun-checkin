import os
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import requests

import checkin


class CheckinTestCase(unittest.TestCase):
    def setUp(self):
        # Tests must neither read real credentials nor make network requests.
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        network = patch.object(
            requests.sessions.Session,
            "request",
            side_effect=AssertionError("Unexpected network request in unit test"),
        )
        network.start()
        self.addCleanup(network.stop)
        logger = patch.object(checkin, "logger")
        self.logger = logger.start()
        self.addCleanup(logger.stop)


class ResponseTests(CheckinTestCase):
    def setUp(self):
        super().setUp()
        self.api = checkin.API("glados.cloud", 1)
        self.addCleanup(self.api.close)

    def test_code_zero_is_confirmed_success_and_preserves_api_message(self):
        result = self.api._parse_checkin(
            {"code": 0, "message": "checkin! Got 7 Points!", "points": 7}
        )
        self.assertEqual(checkin.CheckinStatus.SUCCESS, result["code"])
        self.assertEqual(0, result["api_code"])
        self.assertEqual("checkin! Got 7 Points!", result["message"])
        self.assertEqual("7", result["points"])
        self.assertIn("Got 7 Points", str(self.logger.info.call_args_list))

    def test_unknown_code_one_is_failure_instead_of_assumed_repeat(self):
        result = self.api._parse_checkin({"code": 1, "message": "Please login"})
        self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])
        self.assertEqual(1, result["api_code"])
        self.assertEqual("Please login", result["message"])

    def test_unverified_repeat_message_does_not_confirm_checkin(self):
        result = self.api._parse_checkin({"code": 1, "message": "Already checked in"})
        self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])

    def test_network_failure_is_explicit_failure_with_no_api_code(self):
        result = self.api._parse_checkin(None)
        self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])
        self.assertIsNone(result["api_code"])

    def test_exception_wrapper_preserves_explicit_failure_code(self):
        with patch.object(self.api, "_checkin_attempt", side_effect=ValueError("bad response")):
            result = self.api.checkin("test-cookie")
        self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])
        self.assertIsNone(result["api_code"])
        self.assertIn("ValueError", result["message"])
        self.assertNotIn("bad response", result["message"])

    def test_device_mismatch_retries_using_reported_platform(self):
        with patch.object(
            self.api,
            "_checkin_attempt",
            side_effect=[
                {"code": 4, "reason": "device-mismatch", "loginDevice": "Windows"},
                {"code": 0, "message": "checkin! Got 7 Points!"},
            ],
        ) as attempt:
            result = self.api.checkin("test-cookie")
        self.assertEqual(checkin.CheckinStatus.SUCCESS, result["code"])
        self.assertEqual(checkin.PLATFORM_UA["Windows"], attempt.call_args.args[3])

    def test_points_reads_official_current_points_field(self):
        response = Mock(ok=True)
        response.json.return_value = {"code": 0, "currentPoints": 32}
        with patch.object(self.api, "_make_request", return_value=response):
            self.assertEqual(("32 积分", 32), self.api.get_points("test-cookie"))

    def test_points_keeps_legacy_points_compatibility(self):
        response = Mock(ok=True)
        response.json.return_value = {"code": 0, "points": 25}
        with patch.object(self.api, "_make_request", return_value=response):
            self.assertEqual(("25 积分", 25), self.api.get_points("test-cookie"))

    def test_malformed_success_code_does_not_confirm_checkin(self):
        for code in (False, True, "0", None):
            with self.subTest(code=code):
                result = self.api._parse_checkin({"code": code})
                self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])

    def check_code_one_with_history(self, payload):
        response = Mock(ok=True)
        response.json.return_value = payload
        with patch.object(
            self.api, "_checkin_attempt", return_value={"code": 1, "message": "unknown"}
        ), patch.object(self.api, "_make_request", return_value=response), patch.object(
            checkin, "datetime"
        ) as clock:
            clock.now.return_value = datetime(2026, 10, 18, 0, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
            result = self.api.checkin("test-cookie")
            clock.now.assert_called_with(ZoneInfo("Asia/Shanghai"))
            return result

    def test_code_one_requires_today_checkin_history_for_verified_repeat(self):
        result = self.check_code_one_with_history(
            {"code": 0, "history": [{"business": "system:checkin", "detail": "2026-10-18"}]}
        )
        self.assertEqual(checkin.CheckinStatus.REPEAT, result["code"])
        self.assertEqual(1, result["api_code"])

    def test_code_one_rejects_missing_yesterday_unrelated_or_unauthenticated_history(self):
        payloads = [
            {"code": 0},
            {"code": 0, "history": []},
            {"code": 0, "history": [{"business": "system:checkin", "detail": "2026-10-17"}]},
            {"code": 0, "history": [{"business": "system:other", "detail": "2026-10-18"}]},
            {"code": 1, "history": [{"business": "system:checkin", "detail": "2026-10-18"}]},
            {"code": 0, "history": "invalid"},
            {"code": 0, "history": [None, "invalid"]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(
                    checkin.CheckinStatus.FAILURE, self.check_code_one_with_history(payload)["code"]
                )

    def test_code_one_cannot_be_verified_after_network_failure(self):
        with patch.object(
            self.api, "_checkin_attempt", return_value={"code": 1, "message": "unknown"}
        ), patch.object(self.api, "_make_request", return_value=None):
            self.assertEqual(checkin.CheckinStatus.FAILURE, self.api.checkin("test-cookie")["code"])

    def test_invalid_history_json_preserves_original_business_code(self):
        response = Mock(ok=True)
        response.json.side_effect = ValueError("invalid JSON")
        with patch.object(
            self.api, "_checkin_attempt", return_value={"code": 1, "message": "unknown"}
        ), patch.object(self.api, "_make_request", return_value=response):
            result = self.api.checkin("test-cookie")
        self.assertEqual(checkin.CheckinStatus.FAILURE, result["code"])
        self.assertEqual(1, result["api_code"])


class MainTests(CheckinTestCase):
    def run_main(self, codes, cookies="test-cookie"):
        def domain_result(cookie, cookie_index, domain):
            return checkin.CheckinResult(cookie_index, domain, code=next(results))

        results = iter(codes)
        with patch.dict(os.environ, {"GLADOS_COOKIES": cookies}), patch.object(
            checkin.Checker, "_checkin_on_domain", side_effect=domain_result
        ), patch.object(checkin.PushService, "send", return_value=False):
            return checkin.main()

    def test_all_domain_failures_return_nonzero(self):
        failure = checkin.CheckinStatus.FAILURE
        self.assertEqual(1, self.run_main([failure, failure]))

    def test_one_confirmed_domain_is_enough_for_a_cookie(self):
        self.assertEqual(
            0, self.run_main([checkin.CheckinStatus.FAILURE, checkin.CheckinStatus.SUCCESS])
        )

    def test_each_cookie_must_have_a_confirmed_domain(self):
        success, failure = checkin.CheckinStatus.SUCCESS, checkin.CheckinStatus.FAILURE
        self.assertEqual(1, self.run_main([success, failure, failure, failure], "one&two"))

    def test_each_cookie_confirmed_returns_zero(self):
        success, failure = checkin.CheckinStatus.SUCCESS, checkin.CheckinStatus.FAILURE
        self.assertEqual(0, self.run_main([success, failure, failure, success], "one&two"))

    def test_verified_repeat_satisfies_a_cookie(self):
        repeat = checkin.CheckinStatus.REPEAT
        self.assertEqual(0, self.run_main([repeat, repeat]))

    def test_missing_cookies_return_nonzero(self):
        with patch.object(checkin.PushService, "send", return_value=False):
            self.assertEqual(1, checkin.main())

    def test_configuration_exception_returns_nonzero_without_broken_push_config(self):
        with patch.object(checkin.Config, "_load_config", side_effect=ValueError("bad config")):
            self.assertEqual(1, checkin.main())

    def test_notification_error_does_not_change_confirmed_checkin_exit_status(self):
        result = checkin.CheckinResult(1, "glados.cloud", code=checkin.CheckinStatus.SUCCESS)
        with patch.dict(os.environ, {"GLADOS_COOKIES": "test-cookie"}), patch.object(
            checkin.Checker, "_checkin_on_domain", return_value=result
        ), patch.object(checkin.PushService, "send", side_effect=RuntimeError("push down")):
            self.assertEqual(0, checkin.main())

    def test_notification_error_does_not_hide_failed_checkin_exit_status(self):
        result = checkin.CheckinResult(1, "glados.cloud")
        with patch.dict(os.environ, {"GLADOS_COOKIES": "test-cookie"}), patch.object(
            checkin.Checker, "_checkin_on_domain", return_value=result
        ), patch.object(checkin.PushService, "send", side_effect=RuntimeError("push down")):
            self.assertEqual(1, checkin.main())

    def test_main_entrypoint_exits_nonzero_when_no_cookies_exist(self):
        with patch("logging_config.init_logger", return_value=self.logger):
            with self.assertRaises(SystemExit) as exit_context:
                runpy.run_path(str(Path(checkin.__file__)), run_name="__main__")
        self.assertEqual(1, exit_context.exception.code)


class SummaryTests(CheckinTestCase):
    def test_domain_result_preserves_response_code_message_and_points(self):
        config = checkin.Config()
        checker = checkin.Checker(config)
        with patch.object(checkin, "API") as api_cls:
            api = api_cls.return_value.__enter__.return_value
            api.get_status.return_value = ("10 天", 0)
            api.checkin.return_value = {
                "code": checkin.CheckinStatus.SUCCESS,
                "api_code": 0,
                "status": "签到成功",
                "message": "checkin! Got 7 Points!",
                "points": "7",
            }
            api.get_points.return_value = ("32 积分", 32)
            result = checker._checkin_on_domain("test-cookie", 1, "glados.cloud")
        self.assertEqual(0, result.api_code)
        self.assertEqual("checkin! Got 7 Points!", result.message)
        self.assertEqual("7", result.points)

    def test_summary_includes_business_code_and_message_without_verbose(self):
        checker = checkin.Checker(checkin.Config())
        result = checkin.CheckinResult(1, "glados.cloud")
        result.api_code = 4
        result.message = "device mismatch"
        checker.results.append(result)
        _, content, log_content = checker.format_results()
        for text in (content, log_content):
            self.assertIn("api_code=4", text)
            self.assertIn("device mismatch", text)

    def test_main_writes_reviewable_step_summary(self):
        result = checkin.CheckinResult(1, "glados.cloud")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.md"
            with patch.dict(
                os.environ,
                {"GLADOS_COOKIES": "test-cookie", "GITHUB_STEP_SUMMARY": str(summary_path)},
            ), patch.object(checkin.Checker, "_checkin_on_domain", return_value=result), patch.object(
                checkin.PushService, "send", return_value=False
            ):
                checkin.main()
            self.assertIn("签到", summary_path.read_text())
            self.assertIn("api_code", summary_path.read_text())


class RedactionTests(CheckinTestCase):
    def test_invalid_cookie_header_is_not_exposed_in_logs_or_summary(self):
        secret = "synthetic-cookie-secret"
        cookie = f"session={secret}\ninvalid-header"
        # Preparing a request validates the header without opening any connection.
        with self.assertRaises(requests.exceptions.InvalidHeader) as invalid_header:
            requests.Request("GET", "https://example.invalid", headers={"Cookie": cookie}).prepare()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.md"
            with patch.dict(
                os.environ, {"GLADOS_COOKIES": cookie, "GITHUB_STEP_SUMMARY": str(summary_path)}
            ), patch.object(
                requests.sessions.Session, "request", side_effect=invalid_header.exception
            ), patch.object(checkin.PushService, "send", return_value=False):
                self.assertEqual(1, checkin.main())
            self.assertNotIn(secret, str(self.logger.mock_calls))
            self.assertNotIn(secret, summary_path.read_text())

    def test_decorated_exception_does_not_expose_cookie_value(self):
        with checkin.API("glados.cloud", 1) as api:
            with patch.object(api, "_checkin_attempt", side_effect=ValueError("synthetic-secret")):
                result = api.checkin("test-cookie")
        self.assertNotIn("synthetic-secret", str(self.logger.mock_calls))
        self.assertNotIn("synthetic-secret", result["message"])

    def test_main_exception_does_not_expose_credentials_in_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.md"
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_path)}), patch.object(
                checkin.Config, "_load_config", side_effect=ValueError("synthetic-secret")
            ):
                self.assertEqual(1, checkin.main())
            self.assertNotIn("synthetic-secret", str(self.logger.mock_calls))
            self.assertNotIn("synthetic-secret", summary_path.read_text())

    def test_non_success_http_response_body_is_not_logged(self):
        response = Mock(ok=False, status_code=403, text="synthetic-secret")
        with checkin.API("glados.cloud", 1) as api:
            with patch.object(api.session, "get", return_value=response):
                self.assertIsNone(api._make_request("https://glados.cloud/api/user/status", "GET"))
        self.assertNotIn("synthetic-secret", str(self.logger.mock_calls))
        self.assertIn("403", str(self.logger.mock_calls))


class PushTests(CheckinTestCase):
    def setUp(self):
        super().setUp()
        self.config = checkin.Config()
        self.config.push_key = "synthetic-push-secret"

    def test_push_request_has_finite_timeout_and_checks_http_status(self):
        response = Mock()
        response.json.return_value = {"content": {"result": [json.dumps({"success": "ok"})]}}
        with patch.object(requests, "get", return_value=response) as get:
            self.assertTrue(checkin.PushService(self.config).send("title", "content"))
        self.assertEqual((10, 20), get.call_args.kwargs["timeout"])
        self.assertEqual(self.config.push_key, get.call_args.kwargs["params"]["pushkey"])
        response.raise_for_status.assert_called_once_with()

    def test_push_false_result_is_reported_as_failure(self):
        response = Mock()
        response.json.return_value = {"content": {"result": []}}
        with patch.object(requests, "get", return_value=response):
            self.assertFalse(checkin.PushService(self.config).send("title", "content"))
        self.assertNotIn("推送通知发送成功", str(self.logger.info.call_args_list))

    def test_push_timeout_does_not_expose_push_key(self):
        with patch.object(
            requests, "get", side_effect=requests.exceptions.Timeout(self.config.push_key)
        ):
            self.assertFalse(checkin.PushService(self.config).send("title", "content"))
        self.assertNotIn(self.config.push_key, str(self.logger.mock_calls))


if __name__ == "__main__":
    unittest.main()
