"""Stage 6 CLI dispatch tests (no datasets, no protected outcomes)."""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cftr_theratype import cli


def test_help(capsys):
    with pytest.raises(SystemExit) as ex:
        cli.main(['--help'])
    assert ex.value.code == 0
    assert 'transfer' in capsys.readouterr().out


def test_stage3_legacy_alias(monkeypatch, tmp_path):
    from cftr_theratype.experiments import feature_benchmark as b
    calls = []
    monkeypatch.setattr(b, 'run', lambda root, **kwargs: calls.append((root, kwargs)))
    assert cli.main(['benchmark', '--target', 'corrector', '--root', str(tmp_path)]) == 0
    assert calls[0][1]['target'] == 'corrector'
    assert cli.main(['benchmark', 'run', '--target', 'total', '--root', str(tmp_path)]) == 0
    assert calls[1][1]['target'] == 'total'


def test_old_compare_alias(monkeypatch):
    from cftr_theratype.experiments import feature_benchmark as b
    calls = []
    monkeypatch.setattr(b, 'compare_oof', lambda *a, **kw: calls.append((a, kw)))
    assert cli.main(['compare', '--new', 'new.csv', '--reference', 'old.csv']) == 0
    assert str(calls[0][0][0]) == 'new.csv'


def test_no_unimplemented_command():
    with pytest.raises(SystemExit) as ex:
        cli.main(['transfer', 'run', '--config', 'nonexistent.yaml'])
    assert ex.value.code == 2
