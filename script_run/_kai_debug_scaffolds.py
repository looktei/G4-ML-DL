"""Debug: scan dataset for molecules that break MurckoScaffoldSmiles."""
import csv
import sys
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')  # quiet

CSV_PATH = '/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv'

bad = []
total = 0
ok = 0
null_smiles = 0

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        smi = row.get('smiles', '').strip()
        if not smi:
            null_smiles += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append({
                'compound_id': row.get('compound_id', ''),
                'label': row.get('label', ''),
                'smiles': smi,
                'reason': 'MolFromSmiles returned None',
            })
            continue
        try:
            _ = MurckoScaffold.MurckoScaffoldSmiles(mol, includeChirality=True)
            ok += 1
        except Exception as e:
            bad.append({
                'compound_id': row.get('compound_id', ''),
                'label': row.get('label', ''),
                'smiles': smi,
                'reason': f'{type(e).__name__}: {str(e)[:120]}',
            })

print(f'Total rows:    {total}')
print(f'Null smiles:   {null_smiles}')
print(f'OK scaffolds:  {ok}')
print(f'BAD scaffolds: {len(bad)}')
print()
print('--- First 10 bad molecules ---')
for b in bad[:10]:
    print(f"  [{b['label']}] {b['compound_id']}: {b['smiles'][:80]}")
    print(f"      -> {b['reason']}")
