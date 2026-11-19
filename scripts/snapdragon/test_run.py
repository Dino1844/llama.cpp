import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("snapdragon_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MoeCacheConfigurationTests(unittest.TestCase):
    def run_launcher(self, target, options=(), env=None, expected_exit=0):
        argv = ["run.py", "--install-dir", "test-package"]
        if target:
            argv.extend(["--target", target])
        argv.extend(options)
        argv.extend(["--", "llama-cli"])
        with patch.dict(os.environ, env or {}, clear=True), patch.object(sys, "argv", argv), \
                patch.object(runner.platform, "system", return_value="Linux"), \
                patch.object(runner.subprocess, "run") as process, \
                contextlib.redirect_stderr(io.StringIO()):
            process.return_value.returncode = 0
            with self.assertRaises(SystemExit) as result:
                runner.main()
            self.assertEqual(result.exception.code, expected_exit)
        if expected_exit:
            process.assert_not_called()
            return {}
        process.assert_called_once()
        if target is None:
            return process.call_args.kwargs["env"]
        command = process.call_args.args[0][-1]
        return dict(word.split("=", 1) for word in shlex.split(command) if "=" in word)

    def test_environment_reaches_each_transport(self):
        env = {"LLAMA_MOE_CACHE_PREFETCH": "8", "LLAMA_MOE_CACHE_LAYOUT": "window"}
        for target in (None, "android:test-phone", "linux:test-host"):
            with self.subTest(target=target):
                child = self.run_launcher(target, env=env)
                self.assertEqual({name: child.get(name) for name in env}, env)

    def test_cli_overrides_environment_and_preserves_zero_depth(self):
        env = {"LLAMA_MOE_CACHE_PREFETCH": "8", "LLAMA_MOE_CACHE_LAYOUT": "shared"}
        options = ["--moe-cache-prefetch", "0", "--moe-cache-layout", "window"]
        for target in (None, "android:test-phone", "linux:test-host"):
            with self.subTest(target=target):
                child = self.run_launcher(target, options, env)
                self.assertEqual(child.get("LLAMA_MOE_CACHE_PREFETCH"), "0")
                self.assertEqual(child.get("LLAMA_MOE_CACHE_LAYOUT"), "window")

    def test_omitted_configuration_stays_unspecified(self):
        for target in (None, "android:test-phone", "linux:test-host"):
            with self.subTest(target=target):
                child = self.run_launcher(target)
                self.assertNotIn("LLAMA_MOE_CACHE_PREFETCH", child)
                self.assertNotIn("LLAMA_MOE_CACHE_LAYOUT", child)

    def test_invalid_cli_configuration_is_rejected_before_launch(self):
        for option in (["--moe-cache-prefetch", "-1"], ["--moe-cache-prefetch", "9"],
                       ["--moe-cache-prefetch", "bad"], ["--moe-cache-layout", "bad"]):
            with self.subTest(option=option):
                self.run_launcher(None, option, expected_exit=2)


if __name__ == "__main__":
    unittest.main()
