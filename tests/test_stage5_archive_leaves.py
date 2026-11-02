from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('stage5', Path(__file__).resolve().parents[1] / 'scripts/maintenance/stage5_archive_leaves.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_candidate_and_imported_script(tmp_path):
    s = tmp_path / 'scripts'; s.mkdir()
    (tmp_path / 'src').mkdir()
    (s / 'phase99a_leaf.py').write_text('def f():\n    return 1\n')
    (s / 'phase99b_core.py').write_text('def f():\n    return 2\n')
    (s / 'phase99c_user.py').write_text('import phase99b_core\n')
    x = {r['file']: r for r in mod.plan_repo(tmp_path)}
    assert x['scripts/phase99a_leaf.py']['status'] == 'CANDIDATE'
    assert x['scripts/phase99b_core.py']['status'] == 'RETAIN_OR_REVIEW'
    assert 'python_importers' in x['scripts/phase99b_core.py']['reasons']


def test_literal_references_and_file_paths_are_protected(tmp_path):
    s = tmp_path / 'scripts'; s.mkdir()
    (s / 'phase99d_item.py').write_text('print(123)\n')
    (s / 'phase99e_self.py').write_text('print(__file__)\n')
    (s / 'driver.py').write_text('COMMAND = "python scripts/phase99d_item.py"\n')
    x = {r['file']: r for r in mod.plan_repo(tmp_path)}
    assert 'code_or_config_references' in x['scripts/phase99d_item.py']['reasons']
    assert 'file_location_dependency' in x['scripts/phase99e_self.py']['reasons']


def test_no_rewritten_bytes(tmp_path):
    s = tmp_path / 'scripts'; s.mkdir()
    p = s / 'phase99k_plain.py'
    p.write_bytes(b'\r\n\x00unusual original bytes\n')
    rows = mod.plan_repo(tmp_path)
    assert len(rows) == 1
    assert rows[0]['sha256'] == mod.sha256(p)
