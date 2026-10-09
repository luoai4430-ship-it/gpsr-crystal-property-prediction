import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_result_audits():
    module = load_script("audit_release_results.py")
    sft = module.audit_sft(False)
    alignn = module.audit_alignn(False)
    assert sft[2] == 28
    assert alignn[2] == 26


def test_required_release_files():
    module = load_script("verify_release.py")
    module.verify_required_files()
