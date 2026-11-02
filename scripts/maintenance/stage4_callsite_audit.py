#!/usr/bin/env python3
"""Stage 4: read-only import and callsite audit for top-level legacy scripts.

No modifications to scripts, src, tests, data, results, or git state.
Finds static legacy imports and requested functions; flags possible dynamic usage.
Use output for manual, parity-tested dependency migration, NOT automatic deletion.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT_FOLDERS = ("scripts", "src", "tests", "tools")
KEEP_REVIEW = {
    "phase26d_corrector_bottleneck", "phase26g_feature_model_ablation",
    "phase26h_auxiliary_corrector", "phase27d1_biology_ablation",
    "phase27b1_extreme_classifiers", "phase26k_trafficking_preflight",
    "phase31m_three_tracks", "phase32a_adaptive_theratyping",
}


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path, dest: Path) -> dict:
    root = root.resolve()
    if not (root / "scripts").is_dir() or not (root / "src/cftr_theratype").is_dir():
        raise ValueError(f"Not CFTR-Theratype root: {root}")
    # Only direct top-level scripts are subject to the historical-phase inventory.
    scripts = {p.stem: p for p in (root / "scripts").glob("*.py")
               if p.stem.isidentifier()}
    py_files = sorted({p for folder in ROOT_FOLDERS
                       for p in (root / folder).rglob("*.py")
                       if (root / folder).is_dir()
                       and "__pycache__" not in p.parts})
    dep_rows, symbol_rows, dynamic_rows, parse_errors = [], [], [], []
    inbound = defaultdict(set)
    symbols = defaultdict(set)
    indirect_mentions = defaultdict(set)
    for path in py_files:
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            parse_errors.append({"file":rel,"error":str(exc)})
            continue
        module_aliases = {}
        imported_symbols = {}
        local_deps = set()
        # Visit imports anywhere, including inside functions/guards.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0]
                    if root_name in scripts and path != scripts[root_name]:
                        module_aliases[alias.asname or root_name] = root_name
                        local_deps.add((root_name, "import", node.lineno))
            elif isinstance(node, ast.ImportFrom):
                root_name = (node.module or "").split(".")[0]
                if root_name in scripts and path != scripts[root_name]:
                    for alias in node.names:
                        local_deps.add((root_name, "from", node.lineno))
                        imported_symbols[alias.asname or alias.name] = (root_name,alias.name)
                        if alias.name == "*":
                            dynamic_rows.append({"file":rel,"line":node.lineno,
                                                 "reference":root_name,"reason":"star import"})
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"__import__"} and node.args and isinstance(node.args[0], ast.Constant):
                    val = str(node.args[0].value).split(".")[0]
                    if val in scripts:
                        dynamic_rows.append({"file":rel,"line":node.lineno,
                                             "reference":val,"reason":"dynamic __import__"})
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args
                  and isinstance(node.args[0], ast.Constant)):
                val = str(node.args[0].value).split(".")[0]
                if val in scripts:
                    dynamic_rows.append({"file":rel,"line":node.lineno,
                                         "reference":val,"reason":"dynamic importlib"})
        for target,kind,line in sorted(local_deps):
            dep_rows.append({"importer":rel,"module":target,"kind":kind,"line":line})
            inbound[target].add(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in module_aliases:
                    target = module_aliases[node.value.id]
                    symbol_rows.append({"importer":rel,"module":target,
                                        "symbol":node.attr,"line":node.lineno,
                                        "reference_type":"module attribute"})
                    symbols[target].add(node.attr)
            elif isinstance(node, ast.Name) and node.id in imported_symbols:
                target,sym = imported_symbols[node.id]
                if isinstance(node.ctx, ast.Load):
                    symbol_rows.append({"importer":rel,"module":target,
                                        "symbol":sym,"line":node.lineno,
                                        "reference_type":"from-import usage"})
                    symbols[target].add(sym)
        # Flag mentions in code strings even if no static import was recognised.
        for target in scripts:
            if target not in source or path == scripts[target]:
                continue
            if target not in {dep[0] for dep in local_deps}:
                indirect_mentions[target].add(rel)
    all_rows=[]
    for target,p in sorted(scripts.items()):
        text=p.read_text(encoding="utf-8",errors="replace")
        importers=sorted(inbound[target])
        mentions=sorted(indirect_mentions[target])
        review = ("RETAIN_CRITICAL" if target in KEEP_REVIEW
                  else "RETAIN_IMPORTED" if importers
                  else "MANUAL_REVIEW_NO_STATIC_IMPORTS")
        all_rows.append({"script":p.relative_to(root).as_posix(),
                         "lines":len(text.splitlines()),"static_importers":len(importers),
                         "indirect_mentions":len(mentions),
                         "main_guard":bool(re.search(r'if\s+__name__\s*==\s*[\"\']__main__[\"\']',text)),
                         "review_status":review,
                         "importers":";".join(importers),
                         "symbols":";".join(sorted(symbols[target])),
                         "mentions":";".join(mentions)})
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/"legacy_inventory.csv",all_rows,
              ["script","lines","static_importers","indirect_mentions","main_guard",
               "review_status","importers","symbols","mentions"])
    write_csv(dest/"static_imports.csv",dep_rows,["importer","module","kind","line"])
    write_csv(dest/"used_symbols.csv",symbol_rows,
              ["importer","module","symbol","line","reference_type"])
    write_csv(dest/"dynamic_import_flags.csv",dynamic_rows,
              ["file","line","reference","reason"])
    report = {"root":str(root),"n_scripts":len(scripts),"n_scanned_py":len(py_files),
              "n_static_imports":len(dep_rows),"n_symbol_occurrences":len(symbol_rows),
              "n_dynamic_flags":len(dynamic_rows),"parse_errors":parse_errors,
              "top_dependency_hubs":[{"module":r["script"],
                                      "static_importers":r["static_importers"],
                                      "symbols":r["symbols"]}
                                     for r in sorted(all_rows,key=lambda x:-x["static_importers"])[:20]],
              "review_counts":{key:sum(row["review_status"]==key for row in all_rows)
                               for key in sorted({row["review_status"] for row in all_rows})},
              "notes":["This audit is read-only and does not establish deletion safety.",
                       "Dynamic file paths, shell commands and subprocess references may not be statically detectable.",
                       "Historical scripts may be reproducibility assets even if not imported.",
                       "Verify source checkpoint, preserve Git snapshot, then migrate callers with parity tests."]}
    (dest/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("="*72)
    print("CFTR STAGE4 | LEGACY IMPORT & SYMBOL MIGRATION AUDIT")
    print("="*72)
    print("Scripts:",len(scripts),"scanned Python files:",len(py_files))
    print("Static imports:",len(dep_rows),"symbol occurrences:",len(symbol_rows))
    print("Review categories:",report["review_counts"])
    print("\nTOP DEPENDENCY HUBS:")
    for item in report["top_dependency_hubs"][:12]:
        print(f'{item["static_importers"]:3d} importers  {item["module"]}: {item["symbols"][:100]}')
    print("\nSaved:",dest)
    print("NO FILES MOVED, MODIFIED OR DELETED")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--outdir",type=Path,
                        default=Path("results/maintenance/refactor_audit/stage4"))
    args=parser.parse_args(argv)
    audit(args.root,args.outdir)


if __name__ == "__main__":
    main()
