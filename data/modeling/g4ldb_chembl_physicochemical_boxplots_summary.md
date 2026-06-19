# Physicochemical Property Box Plot Summary

- Input rows: 9,344
- Group 1: 4,672 G4LDB positive molecules
- Group 2: 4,672 ChEMBL assumed-negative molecules
- Statistical test: two-sided Mann-Whitney U test for each property
- Multiple testing correction: Benjamini-Hochberg FDR across the five properties
- Plot note: y axes are clipped at the 99.5th percentile only when extreme values would flatten the boxes; statistics use all molecules.
- Figure: `data/modeling/g4ldb_chembl_physicochemical_boxplots.png`
- Statistics CSV: `data/modeling/g4ldb_chembl_physicochemical_stats.csv`

## Results

| Property | Group 1 median | Group 2 median | Difference | p-value | q-value | Sig. | Cliff's delta |
|---|---:|---:|---:|---:|---:|:---:|---:|
| Molecular weight | 520.586 | 500.629 | 19.957 | 1.5e-12 | 3.8e-12 | **** | 0.084 |
| AlogP | 4.269 | 4.209 | 0.060 | 0.012 | 0.020 | * | 0.030 |
| H-bond donors | 2.000 | 2.000 | 0.000 | 0.287 | 0.287 | ns | -0.012 |
| H-bond acceptors | 6.000 | 6.000 | 0.000 | 0.024 | 0.030 | * | -0.027 |
| TPSA | 81.240 | 95.400 | -14.160 | 1.3e-23 | 6.5e-23 | **** | -0.120 |
