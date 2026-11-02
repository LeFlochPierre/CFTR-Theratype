"""AST migration must leave unrelated code, functions, and call signatures intact."""
from pathlib import Path
import importlib.util
import ast

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'scripts/maintenance/stage6_migrate_hubs.py'
spec = importlib.util.spec_from_file_location('stage6_migrate_hubs', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_wrapper_rewrite():
    s = '# foo\ndef foo(x):\n    return x + 2\n\ndef bar():\n    return foo(2)\n'
    after = module.rewrite_source(s, {'foo': 'def foo(x):\n    return x * 3\n'})
    assert '# foo' in after
    assert 'def bar():\n    return foo(2)' in after
    ns = {}
    exec(compile(after, '<test>', 'exec'), ns)
    assert ns['bar']() == 6


def test_all_declared_wrappers_compile():
    for rel, wrappers in module.WRAPPERS.items():
        assert len(wrappers)
        for s in wrappers.values():
            assert len(ast.parse(s).body) == 1
            assert isinstance(ast.parse(s).body[0], ast.FunctionDef)
