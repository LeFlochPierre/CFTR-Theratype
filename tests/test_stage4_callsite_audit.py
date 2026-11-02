from pathlib import Path
import importlib.util


def test_static_imports_and_symbols(tmp_path):
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "src/cftr_theratype").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts/phase26d_corrector_bottleneck.py").write_text(
        "def load_train(): pass\ndef model(): pass\n")
    (root / "scripts/phase26h_auxiliary_corrector.py").write_text(
        "import phase26d_corrector_bottleneck as legacy\n"
        "from phase26d_corrector_bottleneck import model as build\n"
        "def run():\n    return legacy.load_train(), build()\n")
    (root / "scripts/phase32a_adaptive_theratyping.py").write_text(
        "print('hello')\n")
    script = Path(__file__).parents[1]/"scripts/maintenance/stage4_callsite_audit.py"
    spec = importlib.util.spec_from_file_location("stage4_callsite",script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    report=m.audit(root,root/"out")
    assert report["n_scripts"]==3
    assert report["n_static_imports"]==2
    content=(root/"out/used_symbols.csv").read_text()
    assert "load_train" in content
    assert "model" in content
    assert "phase26d_corrector_bottleneck" in content
    assert "NO_FILE" not in content
