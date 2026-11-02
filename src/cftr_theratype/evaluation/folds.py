"""Frozen, protein-position-disjoint historical OOF splits."""

def from_historical_oof(oof, expected_keys, position_by_key):
    required = {'repeat', 'fold', 'model', 'variant_key', 'protein_position'}
    if required - set(oof):
        raise ValueError(f'Missing OOF columns: {sorted(required-set(oof))}')
    expected = set(expected_keys)
    reference = None
    for name in sorted(oof['model'].dropna().astype(str).unique()):
        t = oof.loc[oof['model'].astype(str).eq(name)]
        if not len(t):
            continue
        ok = True
        for _, g in t.groupby('repeat', dropna=False):
            if set(g.variant_key) != expected or g.variant_key.duplicated().any():
                ok = False
                break
            if (g.groupby('protein_position').fold.nunique() > 1).any():
                ok = False
                break
        if ok:
            reference = name
            break
    if reference is None:
        raise ValueError('No complete position-disjoint OOF reference')
    selected = oof.loc[oof['model'].astype(str).eq(reference)]
    splits = []
    for (repeat, fold), held in selected.groupby(['repeat', 'fold'], sort=True):
        test = set(held.variant_key)
        train = expected - test
        held_positions = set(held.protein_position)
        train_positions = {position_by_key[k] for k in train}
        if not train or not test or not held_positions.isdisjoint(train_positions):
            raise ValueError(f'Position leakage in repeat={repeat}, fold={fold}')
        splits.append((repeat, fold, train, test))
    return splits, reference
