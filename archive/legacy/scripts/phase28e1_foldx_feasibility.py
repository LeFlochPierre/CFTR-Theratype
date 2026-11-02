#!/usr/bin/env python3
"""Phase 28E-1 | Audited FoldX mutation feasibility from CFTR TRAIN features.

No response outcome values, validation or test labels, or predictive model training.
Inputs: Phase28C TRAIN-only mutation identifiers (210 records, 206 substitutions)
        UniProt P13569 canonical FASTA and deposited 8EIG/7SV7 PDB structures.
Outputs: residue-level manifest, protein-only coordinate templates, and separate
         FoldX individual_list.txt files (one unique mutation per line).

IMPORTANT: structures are corrected/drug-bound experimental conformations, not
native untreated wild-type CFTR; 8EIG is F508del and both contain E1371Q.
FoldX energies from these templates must not be called VTI rescue energies.
No FoldX executable, license, or prediction is required for this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqIO import read as read_sequence
from Bio.SeqUtils import seq1

AA = 'ACDEFGHIKLMNPQRSTVWY'
REGEX = re.compile(rf'^([{AA}])(\d{{1,4}})([{AA}])$')
N = 1480
PDBS = ('7SV7', '8EIG')
DOMAINS = (('TMD1',81,365),('NBD1',423,646),('R',654,831),('TMD2',859,1155),('NBD2',1210,1443))


def domain(position: int) -> str:
    for name, lo, hi in DOMAINS:
        if lo <= position <= hi:
            return name
    return 'other'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_variant(code: str, canonical: str) -> tuple[str, int, str]:
    m = REGEX.fullmatch(str(code).strip().upper())
    if not m:
        raise ValueError(f'Expected one missense HGVS protein code such as I203T, got {code!r}')
    w, p, a = m.group(1), int(m.group(2)), m.group(3)
    if not 1 <= p <= len(canonical):
        raise ValueError(f'{code}: out of canonical sequence bounds')
    if canonical[p - 1] != w:
        raise ValueError(f'{code}: claimed WT {w} but UniProt P13569 site {p} is {canonical[p-1]}')
    if w == a:
        raise ValueError(f'{code}: synonymous protein substitution is not supported')
    return w, p, a


def load_variants(path: Path, canonical: str, require_frozen: bool=True) -> pd.DataFrame:
    # Only the three safe, label-free columns are accessed.
    data = pd.read_csv(path, dtype={'variant_key': str}, usecols=['variant_key','protein_position','mutation'])
    data['variant_key'] = data.variant_key.astype(str)
    if data.variant_key.isna().any() or data.variant_key.duplicated().any():
        raise ValueError('Missing or repeated variant_key')
    data['protein_position'] = pd.to_numeric(data.protein_position, errors='raise').astype(int)
    codes = data.mutation.astype(str).str.strip().str.upper().tolist()
    for code, position in zip(codes, data.protein_position):
        _, p, _ = parse_variant(code, canonical)
        if p != position:
            raise ValueError(f'{code} differs from protein_position={position}')
    data['mutation'] = codes
    if require_frozen and (len(data)!=210 or data.mutation.nunique()!=206 or data.protein_position.nunique()!=168):
        raise ValueError(f'Unexpected TRAIN cardinality: records={len(data)} substitutions={data.mutation.nunique()} positions={data.protein_position.nunique()}')
    if (data.groupby('mutation').protein_position.nunique() != 1).any():
        raise ValueError('Protein mutation maps to multiple sequence positions')
    return data


class ChainOnly(Select):
    def accept_model(self, model):
        return model.id == 0
    def accept_chain(self, chain):
        return chain.id == 'A'
    def accept_residue(self, residue):
        return residue.id[0] == ' '
    def accept_atom(self, atom):
        return atom.element not in ('H','D')


def read_structure(path: Path, canonical: str, min_residues: int=900):
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    models = list(structure.get_models())
    if len(models)!=1:
        raise ValueError(f'{path}: expected exactly one model, found {len(models)}')
    model = models[0]
    if 'A' not in model:
        raise ValueError(f'{path}: author chain A missing; manual investigation required')
    chain = model['A']
    positions = {}
    mismatches = []
    for r in chain:
        if r.id[0] != ' ':
            continue
        p = int(r.id[1])
        if not 1 <= p <= len(canonical):
            continue
        if r.id[2] != ' ':
            raise ValueError(f'{path}: insertion code at site {p}; residue numbering unsafe')
        aa = seq1(r.resname, custom_map={'UNK':'X'})
        if aa not in AA:
            raise ValueError(f'{path}: non-standard residue {r.resname} at {p}')
        if p in positions:
            raise ValueError(f'{path}: duplicated author residue number {p}')
        positions[p] = {'aa':aa, 'backbone_complete':all(a in r for a in ('N','CA','C','O')),
                        'n_atoms':len(list(r.get_atoms()))}
        if aa != canonical[p - 1]:
            mismatches.append([p, aa, canonical[p-1]])
    if len(positions) < min_residues:
        raise ValueError(f'{path}: only {len(positions)} mapped residues, expect >= {min_residues}')
    if (len(positions)-len(mismatches))/len(positions) < 0.99:
        raise ValueError(f'{path}: less than 99% canonical residue agreement')
    if {int(x[0]) for x in mismatches} - {1371}:
        raise ValueError(f'{path}: unexpected engineered/noncanonical sites {mismatches[:10]}')
    if len(mismatches)!=1 or mismatches[0]!=[1371, 'Q', 'E']:
        raise ValueError(f'{path}: expected documented E1371Q only; observed {mismatches}')
    if path.stem.upper()=='8EIG' and 508 in positions:
        raise ValueError('8EIG should be F508del; position 508 unexpectedly present')
    return structure, positions, mismatches


def eligible_mutation(w: str, p: int, positions: dict, canonical: str) -> tuple[bool,str,str]:
    info=positions.get(p)
    if info is None:
        return False, 'unmodelled_site', ''
    aa=info['aa']
    if aa!=w or aa!=canonical[p-1]:
        return False, 'engineered_or_mismatched_reference', aa
    if not info['backbone_complete']:
        return False, 'incomplete_backbone', aa
    return True, 'eligible_for_foldx_preflight', aa


def make_manifests(variants: pd.DataFrame, sites: dict, canonical: str) -> pd.DataFrame:
    rows=[]
    for name,positions in sites.items():
        for record in variants.itertuples(index=False):
            wt,p,mut = parse_variant(record.mutation,canonical)
            ok,reason,observed=eligible_mutation(wt,p,positions,canonical)
            rows.append({'pdb':name,'variant_key':record.variant_key,
                         'mutation':record.mutation,'protein_position':p,
                         'domain':domain(p),'canonical_wt':wt,'pdb_wt':observed,
                         'mutant_aa':mut,'eligible':int(ok), 'status':reason,
                         'foldx_mutation':f'{wt}A{p}{mut};' if ok else '',
                         'template_engineered_E1371Q':1,
                         'template_F508del':int(name=='8EIG')})
    return pd.DataFrame(rows)


def self_test():
    seq='A'*1480
    seq=seq[:202]+'I'+seq[203:]
    assert parse_variant('I203T',seq)==('I',203,'T')
    assert parse_variant('A46V',seq)==('A',46,'V')
    try:
        parse_variant('I204T',seq)
        raise AssertionError('Invalid WT check not enforced')
    except ValueError:
        pass
    site={203:{'aa':'I','backbone_complete':True,'n_atoms':4},
          46:{'aa':'A','backbone_complete':False,'n_atoms':2},
          1371:{'aa':'Q','backbone_complete':True,'n_atoms':9}}
    assert eligible_mutation('I',203,site,seq) == (True,'eligible_for_foldx_preflight','I')
    assert eligible_mutation('A',46,site,seq)[1]=='incomplete_backbone'
    assert eligible_mutation('A',99,site,seq)[1]=='unmodelled_site'
    assert eligible_mutation('A',1371,site,seq)[1]=='engineered_or_mismatched_reference'
    test=pd.DataFrame([{'variant_key':'v1','protein_position':203,'mutation':'I203T'},
                       {'variant_key':'v2','protein_position':46,'mutation':'A46V'}])
    manifest=make_manifests(test, {'7SV7':site},seq)
    assert list(manifest.eligible)==[1,0]
    assert manifest.foldx_mutation.iloc[0]=='IA203T;'
    assert manifest.foldx_mutation.iloc[1]==''
    print('PASS: mutation/reference mapping, FoldX syntax, duplicate-safe manifests, missing-site and backbone gates')


def run(args):
    self_test()
    features=Path(args.features)
    source=Path(args.structures)
    fasta=source/'P13569.fasta'
    for p in (features,fasta):
        if not p.is_file():
            raise FileNotFoundError(p)
    seq=str(read_sequence(str(fasta),'fasta').seq).upper()
    if len(seq)!=N or seq[203-1]!='I' or seq[1032-1]!='Y':
        raise ValueError('Expected UniProt canonical P13569 length=1480, I203, Y1032')
    variants=load_variants(features,seq,require_frozen=not args.no_frozen_check)
    output=Path(args.out)
    if output.exists() and any(output.rglob('individual_list.txt')) and not args.overwrite:
        raise FileExistsError(f'{output} already contains FoldX lists; require --overwrite to replace')
    output.mkdir(parents=True, exist_ok=True)
    positions={}; pdbs={}; provenance={}
    for name in PDBS:
        p=source/(name+'.pdb')
        if not p.is_file():
            raise FileNotFoundError(p)
        model,site,engineered=read_structure(p,seq)
        positions[name]=site
        pdbs[name]=model
        provenance[name]={'source_pdb':str(p.resolve()),'source_sha256':digest(p),
                          'numbered_sites':len(site),'canonical_agreement':(len(site)-len(engineered))/len(site),
                          'engineered':engineered,'template_F508del':name=='8EIG',
                          'conformation':'drug-bound corrected CFTR; bound ligand not retained in protein-only FoldX template'}
    manifest=make_manifests(variants,positions,seq)
    expected=len(variants)*len(PDBS)
    if len(manifest)!=expected or manifest.duplicated(['pdb','variant_key']).any():
        raise RuntimeError('Manifest duplicates/size mismatch')
    manifest.to_csv(output/'phase28e1_foldx_mutation_manifest.csv',index=False)
    (output/'phase28e1_train_substitutions.csv').write_text(
        variants[['variant_key','mutation','protein_position']].to_csv(index=False))
    print('='*114)
    print('PHASE 28E-1 | MUTATION-SPECIFIC STABILITY FEASIBILITY — TRAIN IDENTIFIERS ONLY')
    print('='*114)
    print(f'TRAIN: {len(variants)} records / {variants.mutation.nunique()} unique substitutions / {variants.protein_position.nunique()} positions')
    print('No VTI labels, protected partitions, regression models or FoldX calculations read/executed.')
    print('\nA. COVERAGE BY EXPERIMENTAL STRUCTURE')
    for name in PDBS:
        d=manifest[manifest.pdb==name]
        eligible=d[d.eligible==1]
        unique=eligible[['mutation','foldx_mutation']].drop_duplicates()
        if unique.foldx_mutation.duplicated().any():
            raise RuntimeError(f'Duplicate mutation syntax in {name}')
        dpath=output/name
        dpath.mkdir(parents=True,exist_ok=True)
        (dpath/'individual_list.txt').write_text('\n'.join(unique.foldx_mutation)+'\n')
        io=PDBIO(); io.set_structure(pdbs[name]); io.save(str(dpath/f'CFTR_{name}_chainA_protein_only.pdb'), ChainOnly())
        # Verify that our exported FoldX input preserves the required author numbering.
        clean=PDBParser(QUIET=True).get_structure('clean',str(dpath/f'CFTR_{name}_chainA_protein_only.pdb'))
        cleanres={int(r.id[1]): seq1(r.resname) for r in clean[0]['A'] if r.id[0]==' '}
        for rec in unique.itertuples(index=False):
            w,p,_=parse_variant(rec.mutation,seq)
            if cleanres.get(p)!=w:
                raise ValueError(f'Exported PDB changed {name} site {p}: {cleanres.get(p)} instead of {w}')
        provenance[name]['protein_only_pdb_sha256']=digest(dpath/f'CFTR_{name}_chainA_protein_only.pdb')
        print(f'{name}: {len(eligible)}/{len(d)} records eligible; {len(unique)}/{variants.mutation.nunique()} substitutions; {eligible.protein_position.nunique()}/{variants.protein_position.nunique()} positions')
        print('   Exclusion reasons:',d.status.value_counts().to_dict())
        print('   Prepared template:',dpath/f'CFTR_{name}_chainA_protein_only.pdb')
        print('   Mutation file:',dpath/'individual_list.txt')
        print('   WARNING: E1371Q engineered into template'+(' and F508del in 8EIG' if name=='8EIG' else ''))
    print('\nB. COVERAGE ACROSS BOTH STRUCTURES (record-level)')
    pivot=manifest.pivot(index='variant_key', columns='pdb',values='eligible')
    pairs=pivot.sum(axis=1)
    print(f'Both eligible: {(pairs==2).sum()} | One eligible: {(pairs==1).sum()} | Neither: {(pairs==0).sum()}')
    anyok=manifest[manifest.eligible==1].drop_duplicates('variant_key')
    print('Eligible variants by domain:',anyok.domain.value_counts().to_dict())
    manifest_sha=digest(output/'phase28e1_foldx_mutation_manifest.csv')
    provenance['inputs']={'source_features_sha256':digest(features),'source_fasta_sha256':digest(fasta),
                         'manifest_sha256':manifest_sha,'train_only':True,
                         'purpose':'FoldX feasibility and coordinate QC, NOT stability predictions or VTI model fitting'}
    (output/'phase28e1_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('\nC. INTERPRETATION / NEXT STEP')
    print('These files are FoldX candidate lists, not FoldX results. FoldX requires a licensed executable and PDB repair.')
    print('Use one mutation per individual_list line; the filename must start with individual_list.')
    print('Run RepairPDB on each protein-only structure BEFORE BuildModel; confirm it preserves residue numbering.')
    print('FoldX stability ΔΔG refers to structural context, not vanzacaftor binding or clinical response.')
    print('Do not interpret 8EIG-vs-7SV7 energies as an isolated treatment effect: templates differ in background/conformation.')
    print('PASS Phase 28E-1 audited preparation. Output:',output.resolve())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--features',default='results/biology/phase28c/phase28c_train_biophysics_features.csv')
    ap.add_argument('--structures',default='data/external/cftr_structures')
    ap.add_argument('--out',default='results/biology/phase28e')
    ap.add_argument('--no-frozen-check',action='store_true',help='For synthetic integration tests only')
    ap.add_argument('--overwrite',action='store_true',help='Overwrite previous FoldX input lists after provenance review')
    args=ap.parse_args()
    if args.self_test:self_test()
    else:run(args)
