"""TRAIN-only metadata and eligible auxiliary source loading for Phase31.

No VTI response outcomes are read. Protected VAL/TEST protein positions are
excluded from cross-assay teacher training. This module preserves the precise
original Phase31I.2 filtering and report schema.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.datasets.assay_tables import (
    S5_NAME, S6_NAME, resolve_file, table_aux,
)


def get_sources(args, root):
    root = Path(root)
    paths = {
        'bihler': root / args.bihler,
        's5': resolve_file(args.s5, S5_NAME, root=root),
        's6': resolve_file(args.s6, S6_NAME, root=root),
        'splits2024': root / args.splits2024,
        'vti_meta': root / args.vti_meta,
    }
    for k, v in paths.items():
        if not v.is_file():
            raise FileNotFoundError(f'Required {k} input missing: {v}')
    return paths


def protected_and_eligible_positions(paths):
    """Partition metadata only, never VTI response values."""
    bsplit = pd.read_parquet(paths['splits2024'], columns=['variant_key','hgvs_protein','protein_position','split'])
    if not bsplit.split.astype(str).str.lower().isin(('train','val','test')).all():
        raise ValueError('Unknown Bihler 2024 split label')
    bp = pd.to_numeric(bsplit.protein_position, errors='coerce')
    bsplit = bsplit.assign(protein_position=bp)
    pos2024_train = set(bsplit.loc[bsplit.split.astype(str).str.lower().eq('train'), 'protein_position'].dropna().astype(int))
    pos2024_protected = set(bsplit.loc[bsplit.split.astype(str).str.lower().isin(['val','test']), 'protein_position'].dropna().astype(int))
    if pos2024_train & pos2024_protected:
        raise AssertionError('Bihler 2024 protein positions cross frozen split boundary')
    # The 2026 split metadata can describe several studies, but every VAL/TEST
    # protein position is conservatively barred from all auxiliary teacher fits.
    vm = pd.read_parquet(paths['vti_meta'], columns=['variant_key','protein_position','transfer_split'])
    vc = pd.to_numeric(vm.protein_position, errors='coerce')
    vm = vm.assign(protein_position=vc)
    if not vm.transfer_split.astype(str).str.lower().isin(('train','val','test','excluded')).all():
        raise ValueError('Unrecognised transfer_split in 2026 VTI metadata')
    pos2026_protected = set(vm.loc[vm.transfer_split.astype(str).str.lower().isin(['val','test']), 'protein_position'].dropna().astype(int))
    protected = pos2024_protected | pos2026_protected
    if not pos2024_train:
        raise ValueError('No Bihler 2024 TRAIN protein positions')
    print(f'Protected positions (2024 val/test union 2026 VTI val/test): {len(protected)}')
    print(f'Bihler 2024 TRAIN positions: {len(pos2024_train)}')
    return pos2024_train, protected


def load_auxiliary(paths, eligible_bihler, protected):
    bih = pd.read_parquet(paths['bihler'], columns=['variant_key','hgvs_protein','eti_rescue_pp'])
    s5 = pd.read_csv(paths['s5'])
    s6 = pd.read_csv(paths['s6'])
    tables, audits = {}, []
    for name, src, field, study in (
        ('bihler',bih,'eti_rescue_pp',None),
        ('p289',s5,'eti_rescue_pp','P289'),
        ('u032',s5,'eti_rescue_pp','U032'),
        ('p289_s6_baseline',s6,'baseline_mature_fraction','P289'),
        ('p289_s6_treated',s6,'eti_mature_fraction','P289'),
        ('u032_s6_baseline',s6,'baseline_mature_fraction','U032'),
        ('u032_s6_treated',s6,'eti_mature_fraction','U032')):
        tab, rep = table_aux(src, name, field, study)
        # Canonical point mutations only. Double check no external positions
        # from frozen protected partitions are in this data.
        before = len(tab)
        tab = tab.loc[~tab.pos31i2.isin(protected)].copy()
        if name == 'bihler':
            tab = tab.loc[tab.pos31i2.isin(eligible_bihler)].copy()
        tab = tab.loc[tab.aa31i.map(lambda x: bool(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY]',x)))].copy()
        if tab.empty:
            raise ValueError(f'No eligible auxiliary labels for {name}; see split filtering')
        if tab.aa31i.duplicated().any():
            raise AssertionError(f'Duplicate exact substitution after canonicalisation: {name}')
        assert set(tab.pos31i2.astype(int)).isdisjoint(protected)
        tables[name] = tab.reset_index(drop=True)
        audits.append(dict(assay=name, rows_source=rep['source_rows'],
                           unique_substitutions_before_filter=before,
                           substitutions_eligible=len(tab),positions_eligible=tab.pos31i2.nunique(),
                           rows_removed_protected_or_split=before-len(tab)))
    return tables, pd.DataFrame(audits)
