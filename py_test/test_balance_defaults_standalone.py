"""Pure-Python entry-point defaults without importing the native extension."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "router_args_standalone", ROOT / "py_src/vllm_router/router_args.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
RouterArgs = module.RouterArgs


class BalanceDefaults(unittest.TestCase):
    def test_dataclass_defaults_and_override(self):
        args = RouterArgs()
        self.assertEqual((args.balance_abs_threshold, args.balance_rel_threshold), (0, 1.2))
        args = RouterArgs(balance_abs_threshold=64, balance_rel_threshold=1.5)
        self.assertEqual((args.balance_abs_threshold, args.balance_rel_threshold), (64, 1.5))

    def test_argparse_default_and_override_with_and_without_prefix(self):
        for prefixed in (False, True):
            with self.subTest(prefixed=prefixed):
                parser = argparse.ArgumentParser()
                RouterArgs.add_cli_args(parser, use_router_prefix=prefixed)
                prefix = "router-" if prefixed else ""
                attr = "router_" if prefixed else ""
                args = parser.parse_args([])
                self.assertEqual((getattr(args, attr + "balance_abs_threshold"),
                                  getattr(args, attr + "balance_rel_threshold")), (0, 1.2))
                args = parser.parse_args([f"--{prefix}balance-abs-threshold", "64",
                                          f"--{prefix}balance-rel-threshold", "1.5"])
                self.assertEqual((getattr(args, attr + "balance_abs_threshold"),
                                  getattr(args, attr + "balance_rel_threshold")), (64, 1.5))

    def test_pyo3_and_rust_default_declarations(self):
        # These are source checks; the native signature is checked by Rust/build tests.
        src = (ROOT / "src/lib.rs").read_text()
        self.assertIn("balance_abs_threshold = 0,", src)
        self.assertIn("balance_rel_threshold = 1.2,", src)
        src = (ROOT / "src/policies/mod.rs").read_text()
        self.assertIn("balance_abs_threshold: 0,", src)
        self.assertIn("balance_rel_threshold: 1.2,", src)


if __name__ == "__main__":
    unittest.main()
