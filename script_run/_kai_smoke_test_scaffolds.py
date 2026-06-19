"""Smoke test: extract Murcko scaffolds from 100 random rows of the real dataset."""
import csv
import random
from rdkit import Chem
import rdkit
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

CSV_PATH = '/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv'
N = 100

rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'RDKit {rdkit.__version__}')
print(f'Total rows: {len(rows)}, sampling: {N}')

random.seed(42)
sample = random.sample(rows, N)

ok = 0
fail = 0
errors = []
for i, row in enumerate(sample, 1):
    smi = row['smiles']
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smi)
        ok += 1
        if i <= 5:
            print(f'  [{i:3d}] {row["compound_id"]}: scaffold = {s[:60]!r}')
    except Exception as e:
        fail += 1
        errors.append((row['compound_id'], smi, str(e)[:80]))

print()
print(f'OK:   {ok}/{N}')
print(f'FAIL: {fail}/{N}')
if errors:
    print()
    print('--- First 5 failures ---')
    for cid, smi, err in errors[:5]:
        print(f'  {cid}: {smi[:50]}... → {err}')
else:
    print('🎉 Zero failures!')
