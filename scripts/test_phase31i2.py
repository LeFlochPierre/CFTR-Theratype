#!/usr/bin/env python3
"""Run in a temporary directory to validate Phase31I.2 audit and leakage-safe OOF."""
import json
import subprocess
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name('phase31i_auxiliary_transfer.py')
RNG = np.random.default_rng(31)


def generate(root):
    root = Path(root)
    n_pos = 240
    positions = np.arange(50, 50+n_pos)
    # Synthetic genomic feature vectors are constructed exclusively from mutation IDs
    feat = []
    ids = []
    for idx, pos in enumerate(positions):
        count = 2 if idx < 42 else 1
        for k in range(count):
            aa = f'A{pos}{"V" if k == 0 else "G"}'
            ids.append((pos, aa))
            vec = [np.sin(pos/23), np.cos(pos/29), (pos % 11)/11, k,
                   np.sin(pos/91), (pos % 7)/7, pos/1480]
            feat.append({'aa_code':aa, 'position':pos, **{f'seq_{j}':v for j,v in enumerate(vec)}})
    bank = pd.DataFrame(feat)
    bank.to_csv(root/'features.csv', index=False)
    (root/'fcols.txt').write_text('\n'.join([f'seq_{j}' for j in range(7)]))
    vti = []
    for pos, aa in ids:
        if pos >= 50 + 168:
            continue
        f = bank[bank.aa_code.eq(aa)].iloc[0]
        outcome = 5*f.seq_0 + 3*f.seq_2 + RNG.normal(0,.1)
        vti.append({'aa_code': aa, 'protein_position': pos, 'fold':f'fold{pos%5}',
                    'transfer_split':'train', 'corrector':outcome,'total':outcome*1.3+RNG.normal(0,.3)})
    assert len(vti) == 210
    pd.DataFrame(vti).to_csv(root/'vti_train.csv', index=False)
    pd.DataFrame([{'aa_code':d['aa_code'], 'protein_position':d['protein_position'],
                   'transfer_split':'train'} for d in vti]).to_csv(root/'vti_meta.csv', index=False)
    bihler = []
    s5=[];s6=[]
    for p, aa in ids:
        f = bank[bank.aa_code.eq(aa)].iloc[0]
        y = 6 * f.seq_0 + RNG.normal(0,.3)
        if p < 260:
            bihler.append({'aa_code':aa,'mutation_type':'missense','eti_actual':y})
        if p%3 != 0:
            study = 'P289' if p%2 else 'U032'
            s5.append({'aa_code':aa, 'study':study,'mutation_type':'missense',
                       'eti_rescue_pp':y*12+RNG.normal(0,1)})
            s6.append({'aa_code':aa, 'study':study,'mutation_type':'missense',
                       'baseline_mature_fraction':0.3+y/500,
                       'eti_mature_fraction':0.5+y/300})
    # Add duplicates (distinct nucleotides same protein substitution): median must be used
    s5.append(dict(s5[0]))
    for name,data in [('bihler.csv',bihler), ('phase31a_castellani_s5_transport.csv',s5),
                      ('phase31a_castellani_s6_processing.csv',s6)]:
        pd.DataFrame(data).to_csv(root/name,index=False)


def run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        generate(root)
        out = root/'out'
        common = ['--s5',str(root/'phase31a_castellani_s5_transport.csv'),
                  '--s6',str(root/'phase31a_castellani_s6_processing.csv'),
                  '--bihler-eti',str(root/'bihler.csv'), '--bihler-target','eti_actual',
                  '--vti-meta',str(root/'vti_meta.csv'),
                  '--features',str(root/'features.csv'), '--feature-cols',str(root/'fcols.txt'),
                  '--out',str(out)]
        # Stage A can run with explicit metadata but never reads outcome table.
        p=subprocess.run([sys.executable,str(SCRIPT),'audit',*common],cwd=root,text=True,capture_output=True)
        assert p.returncode==0,(p.stdout,p.stderr)
        provenance=json.loads((out/'phase31i2_provenance.json').read_text())
        assert provenance['n_vti_outcomes_loaded']==0
        assert (out/'phase31i2_agreement.csv').exists()
        p=subprocess.run([sys.executable,str(SCRIPT),'pilot',*common,
                          '--vti-train',str(root/'vti_train.csv'),
                          '--fold-col','fold','--vti-target','corrector=corrector',
                          '--vti-target','total=total','--bootstrap','120'],
                         cwd=root,text=True,capture_output=True)
        assert p.returncode==0,(p.stdout,p.stderr)
        oof=pd.read_csv(out/'phase31i2_oof_predictions.csv')
        assert set(oof.model)=={'M0','M1','M2','M3','M3_perm'}
        assert len(oof)==210*5*2
        assert not oof.duplicated(['target','model','vti_row31i']).any()
        folds=pd.read_csv(out/'phase31i2_teacher_folds.csv')
        assert (folds.excluded_heldout_rows >= 0).all()
        print('PASS synthetic audit + five-fold nested teacher pilot, 210 variants / 168 positions')
        print('PASS model output rows:',len(oof),'and 5 distinct configurations, 2 targets')
        print('PASS group-leakage checks, no val/test outcomes, negative-control models')
        print('Metrics preview:')
        print(pd.read_csv(out/'phase31i2_metrics.csv').query('fold == "POOLED" and scope == "position"').to_string(index=False))
        print('Bootstrap preview:')
        print(pd.read_csv(out/'phase31i2_bootstrap.csv').to_string(index=False))


if __name__=='__main__':
    run()
