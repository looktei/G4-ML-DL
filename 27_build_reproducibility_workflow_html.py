#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Build original full reproducibility-workflow HTML report."""

STEP_ID = '27'
STEP_TITLE = 'Build original full reproducibility-workflow HTML report'
EMBEDDED_SOURCE_NAME = '28_build_reproducibility_workflow_html.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Build a standalone, English, detailed reproducibility-workflow HTML report\nwith inline base64 figures, covering every stage from 31 May 2026 onward.\nDocumentation only: no training/prediction/threshold tuning.\n"""\nimport base64, mimetypes, os, datetime\n\nROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\nOUT = os.path.join(ROOT, "NIPAH_G4_REPRODUCIBILITY_WORKFLOW_FULL_2026-05-31_to_2026-06-03.html")\n\ndef img(path, cap):\n    full = os.path.join(ROOT, path)\n    if not os.path.exists(full):\n        return f\'<figure class="fig"><div class="missing">Image not found: {path}</div><figcaption>{cap}</figcaption></figure>\'\n    mime = mimetypes.guess_type(full)[0] or "image/png"\n    with open(full, "rb") as f:\n        b64 = base64.b64encode(f.read()).decode("ascii")\n    return (f\'<figure class="fig"><img alt="{cap}" src="data:{mime};base64,{b64}"/>\'\n            f\'<figcaption>{cap}</figcaption></figure>\')\n\nSTYLE = """\n:root{--bg:#f7f8fa;--ink:#17212b;--muted:#5d6a78;--line:#d9e0e7;--panel:#fff;--blue:#2457a6;--teal:#1c7c78;--green:#28784d;--amber:#9a6517;--red:#a63a3a;--code:#101722}\n*{box-sizing:border-box}\nbody{margin:0;background:var(--bg);color:var(--ink);font-family:\'Segoe UI\',Arial,Helvetica,sans-serif;line-height:1.6}\nheader{background:#111827;color:#fff;padding:42px 30px 34px;border-bottom:6px solid var(--teal)}\nheader h1{margin:0 0 10px;font-size:30px;line-height:1.18}\nheader p{max-width:1000px;margin:0;color:#d6dde6;font-size:15px}\nheader .meta{margin-top:14px;font-size:13px;color:#9fb0c0}\nmain{width:min(1180px,calc(100% - 40px));margin:24px auto 70px}\nnav{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin-bottom:26px}\nnav b{display:block;margin-bottom:8px;color:var(--blue)}\nnav a{display:inline-block;margin:3px 14px 3px 0;color:var(--blue);text-decoration:none;font-size:14px}\nnav a:hover{text-decoration:underline}\nsection{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin-bottom:24px}\nh2{color:var(--blue);margin:0 0 6px;font-size:22px;border-bottom:2px solid var(--line);padding-bottom:8px}\nh3{color:var(--teal);margin:22px 0 8px;font-size:17px}\nh4{color:var(--ink);margin:16px 0 6px;font-size:15px}\ntable{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}\nth,td{border:1px solid var(--line);padding:7px 10px;text-align:center}\nth{background:#eef2f7}\ntd.l,th.l{text-align:left}\ntr.hl td{background:#eafaf1;font-weight:600}\n.step{display:inline-block;background:var(--teal);color:#fff;border-radius:6px;padding:2px 10px;font-size:13px;margin-right:8px}\n.kpi{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}\n.kpi div{flex:1 1 150px;background:#f3f7fc;border:1px solid var(--line);border-radius:10px;padding:14px}\n.kpi span{display:block;font-size:12px;color:var(--muted)}\n.kpi b{font-size:22px;color:var(--blue)}\n.fig{margin:18px 0;text-align:center}\n.fig img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}\n.fig figcaption{font-size:13px;color:var(--muted);margin-top:6px}\n.missing{padding:30px;border:1px dashed var(--red);color:var(--red);border-radius:8px}\n.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}\n@media(max-width:820px){.grid2{grid-template-columns:1fr}}\n.note{background:#fff8ec;border-left:4px solid var(--amber);padding:12px 16px;border-radius:6px;margin:12px 0;font-size:14px}\n.warn{background:#fdecec;border-left:4px solid var(--red);padding:12px 16px;border-radius:6px;margin:12px 0;font-size:14px}\n.ok{background:#eafaf1;border-left:4px solid var(--green);padding:12px 16px;border-radius:6px;margin:12px 0;font-size:14px}\n.quote{background:#f4f7fb;border-left:4px solid var(--blue);padding:14px 18px;border-radius:6px;font-style:italic;margin:14px 0}\npre{background:var(--code);color:#e6edf3;padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.45}\ncode{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}\npre code{background:transparent;padding:0;color:inherit}\nul,ol{margin:8px 0 8px 4px}\nli{margin:4px 0}\nfooter{color:var(--muted);font-size:13px;text-align:center;padding:20px}\n"""\n\nGEN = datetime.datetime.now().strftime("%Y-%m-%d %H:%M ICT")\n\nhtml = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Nipah G4 Ligand Classifier — Full Reproducibility Workflow</title>\n<style>{STYLE}</style></head><body>\n<header>\n<h1>Nipah Virus G-Quadruplex Ligand Classifier<br>Full Reproducibility Workflow &amp; Practice Guide</h1>\n<p>A detailed, step-by-step record of the complete project from dataset construction through model development, internal validation, BindingDB and PubChem external evaluation, and imbalance-aware robustness analysis. Designed so an independent researcher can re-run and reproduce every stage.</p>\n<div class="meta">Project window: 31 May 2026 → 3 June 2026 · Generated {GEN} · Prepared by Luke for Luktei · All metrics read from real project artifacts · Locked model: <code>gnn_molecular_graph_scaffold_seed42.pt</code></div>\n</header>\n<main>\n<nav><b>Contents</b>\n<a href="#overview">0. Overview</a><a href="#env">1. Environment</a><a href="#s1">2. Dataset</a>\n<a href="#s2">3. Features &amp; Splits</a><a href="#s3">4. Baseline ML</a><a href="#s4">5. Deep Learning</a>\n<a href="#s5">6. Model Selection</a><a href="#s6">7. BindingDB</a><a href="#s7">8. PubChem</a>\n<a href="#s8">9. Imbalance &amp; Robustness</a><a href="#repro">10. Reproduce End-to-End</a>\n<a href="#limit">11. Limitations</a><a href="#concl">12. Conclusions</a>\n</nav>\n\n<section id="overview"><h2>0. Project Overview &amp; Reproducibility Principles</h2>\n<p><b>Objective.</b> Build and validate a controlled machine-learning workflow that learns chemical patterns of known G4-binding ligands (G4LDB) and uses the model to prioritize compounds for Nipah virus G-quadruplex downstream screening.</p>\n<div class="note"><b>Framing discipline.</b> The first training dataset supports a <i>G4LDB-positive vs ChEMBL assumed-negative</i> classifier. It is not described as a definitive true-binder vs true-non-binder model, because the ChEMBL negative class is assumed-negative rather than experimentally confirmed non-G4-binding.</div>\n<h3>Reproducibility rules enforced throughout</h3>\n<ul>\n<li><b>Single locked checkpoint</b> for all external evaluation: <code>data/modeling/deep_learning_gnn/gnn_molecular_graph_scaffold_seed42.pt</code>.</li>\n<li><b>No retraining, no threshold tuning, no post-hoc model selection</b> on any external dataset.</li>\n<li><b>Fixed seed (42)</b> for the primary comparison; 20 seeds for repeated-seed robustness.</li>\n<li><b>Leakage control</b>: exact training overlap removed and nearest-training Tanimoto similarity reported for every external set.</li>\n<li><b>Traceable labels</b>: every external compound carries its evidence tier and label source.</li>\n</ul>\n<div class="kpi">\n<div><span>Dataset</span><b>9,346</b>4,673 : 4,673</div>\n<div><span>Model families</span><b>5</b>RF/MLP/CNN-BiGRU/ChemBERTa/GNN</div>\n<div><span>Locked GNN scaffold ROC-AUC</span><b>0.9780</b>MCC 0.8462</div>\n<div><span>External eval campaigns</span><b>6</b>BindingDB + PubChem</div>\n</div>\n<h3>Integrated workflow (current, all stages)</h3>\n<div class="fig">{img("data/modeling/final_model_comparison/figures_research_summary/nipah_g4_integrated_workflow.png","Integrated workflow — G4 ligand model development & candidate prioritization (31 May – 3 June 2026)")}</div>\n</section>\n\n<section id="env"><h2>1. Environment &amp; Repository Layout</h2>\n<p>The project ships environment specs and an environment checker.</p>\n<pre><code># Recreate the environment\nconda env create -f environment.yml       # or: pip install -r requirements-colab.txt\npython scripts/check_environment.py        # verifies key dependencies (RDKit, torch, sklearn ...)</code></pre>\n<h4>Key directories</h4>\n<table><tr><th class="l">Path</th><th class="l">Contents</th></tr>\n<tr><td class="l"><code>data/g4ldb/</code></td><td class="l">G4LDB positive ligands + preprocessing artifacts</td></tr>\n<tr><td class="l"><code>data/chembl/</code></td><td class="l">ChEMBL assumed-negative decoy selection</td></tr>\n<tr><td class="l"><code>data/modeling/</code></td><td class="l">Binary dataset, baseline + deep-learning training/validation, final comparison</td></tr>\n<tr><td class="l"><code>data/bindingdb*/</code></td><td class="l">BindingDB known-inhibitor, Kd challenge, supplementary control</td></tr>\n<tr><td class="l"><code>data/pubchem_g4_*/</code></td><td class="l">PubChem BioAssay strict + expanded evidence/background</td></tr>\n<tr><td class="l"><code>scripts/</code></td><td class="l">All fetch / preprocess / train / validate / report scripts</td></tr>\n</table>\n</section>\n\n<section id="s1"><h2>2. Stage 1 — Dataset Construction <span class="step">~31 May</span></h2>\n<h4>2.1 Fetch and preprocess G4LDB positives</h4>\n<pre><code>python scripts/fetch_g4ldb.py\npython scripts/preprocess_g4ldb_ml_ready.py</code></pre>\n<h4>2.2 Build ChEMBL assumed-negative decoys</h4>\n<pre><code>python scripts/prepare_chembl_negative_decoys.py</code></pre>\n<ul>\n<li>ChEMBL negatives filtered against G4LDB positives by Morgan fingerprint / Tanimoto.</li>\n<li>Selected negatives kept <b>below the 0.70 similarity threshold</b>.</li>\n</ul>\n<h4>2.3 Freeze the binary dataset (v1)</h4>\n<p>Primary dataset: <code>data/modeling/g4ldb_chembl_binary_dataset.csv</code></p>\n<ul>\n<li>Positive class: <b>4,673</b> preprocessed G4LDB ligands.</li>\n<li>Negative class: <b>4,673</b> ChEMBL assumed-negative decoys.</li>\n<li>Total: <b>9,346</b> molecules; no duplicate canonical SMILES; no positive–negative SMILES overlap; no missing core fields.</li>\n</ul>\n<pre><code>python scripts/visualize_binary_dataset.py\npython scripts/plot_physicochemical_boxplots.py</code></pre>\n<div class="grid2">\n{img("data/modeling/g4ldb_chembl_binary_dataset_overview.png","Binary dataset overview (G4LDB positives vs ChEMBL negatives)")}\n{img("data/modeling/g4ldb_chembl_physicochemical_boxplots.png","Physicochemical boxplots — MW/TPSA show only mild residual bias")}\n</div>\n<div class="note">Bias check: MW and TPSA show mild residual differences, but effect sizes are small — recorded as a caveat rather than a confound that explains the signal.</div>\n</section>\n\n<section id="s2"><h2>3. Stage 2 — Feature Sets &amp; Split Strategies</h2>\n<h3>Feature sets</h3>\n<ul>\n<li><b>A. Physicochemical descriptors</b> (MW, HBD, HBA, AlogP, TPSA) — bias-check baseline.</li>\n<li><b>B. Morgan fingerprint</b> — radius 2, 2048 bits — main classical representation.</li>\n<li><b>C. Combined</b> — Morgan + descriptors — tests added value/bias.</li>\n</ul>\n<h3>Split strategies</h3>\n<ul>\n<li><b>Random stratified split</b> (80/20, stratified) — sanity check / first estimate.</li>\n<li><b>Scaffold / similarity-aware split</b> — estimates generalization to new scaffolds.</li>\n</ul>\n<div class="note">Interpretation rule: if random performs well but scaffold drops strongly, the model may be memorizing scaffold families. Scaffold split is therefore prioritized for drug-discovery interpretation.</div>\n</section>\n\n<section id="s3"><h2>4. Stage 3 — Baseline ML Training &amp; Validation</h2>\n<pre><code>python scripts/train_baseline_g4_classifier.py\npython scripts/validate_baseline_g4_classifier.py</code></pre>\n<p>Reported metrics: ROC-AUC, PR-AUC, MCC, balanced accuracy, Brier; plus calibration and applicability-domain diagnostics.</p>\n<div class="grid2">\n{img("data/modeling/baseline_training/baseline_roc_auc_comparison.png","Baseline ROC-AUC comparison across feature sets")}\n{img("data/modeling/baseline_training/baseline_pr_auc_comparison.png","Baseline PR-AUC comparison")}\n{img("data/modeling/baseline_training/best_baseline_roc_pr_curves.png","Best baseline — ROC/PR curves")}\n{img("data/modeling/baseline_training/best_baseline_confusion_matrices.png","Best baseline — confusion matrices")}\n{img("data/modeling/baseline_validation/baseline_calibration_curves.png","Baseline — calibration curves")}\n{img("data/modeling/baseline_validation/baseline_applicability_domain.png","Baseline — applicability domain")}\n{img("data/modeling/baseline_validation/baseline_repeated_split_robustness.png","Baseline — repeated-split robustness")}\n{img("data/modeling/baseline_validation/baseline_rf_top_morgan_bits.png","RF — most informative Morgan bits")}\n</div>\n</section>\n\n<section id="s4"><h2>5. Stage 4 — Deep Learning Models</h2>\n<p>Four deep-learning representations were trained with matched seeds and splits.</p>\n<pre><code>python scripts/train_mlp_morgan_g4_classifier.py        # Morgan MLP\npython scripts/train_smiles_sequence_g4_classifier.py   # SMILES CNN-BiGRU\npython scripts/train_chemberta_g4_classifier.py         # ChemBERTa-10M\npython scripts/train_gnn_molecular_graph_g4_classifier.py  # Molecular Graph GNN (locked)</code></pre>\n<h3>Locked Molecular Graph GNN diagnostics</h3>\n<div class="grid2">\n{img("data/modeling/deep_learning_gnn/gnn_molecular_graph_learning_curves.png","GNN — learning curves")}\n{img("data/modeling/deep_learning_gnn/gnn_molecular_graph_primary_roc_pr_curves.png","GNN — primary ROC/PR curves")}\n{img("data/modeling/deep_learning_gnn/gnn_molecular_graph_primary_confusion_matrices.png","GNN — confusion matrices")}\n{img("data/modeling/deep_learning_gnn/gnn_molecular_graph_calibration_curves.png","GNN — calibration curves")}\n</div>\n<div class="fig">{img("data/modeling/deep_learning_gnn/gnn_molecular_graph_repeated_split_robustness.png","GNN — repeated-split robustness (20 seeds)")}</div>\n<h4>Other deep-learning families (representative diagnostics)</h4>\n<div class="grid2">\n{img("data/modeling/deep_learning_mlp/mlp_morgan_primary_roc_pr_curves.png","Morgan MLP — ROC/PR")}\n{img("data/modeling/deep_learning_smiles/smiles_sequence_primary_roc_pr_curves.png","SMILES CNN-BiGRU — ROC/PR")}\n{img("data/modeling/deep_learning_chemberta/chemberta_primary_roc_pr_curves.png","ChemBERTa-10M — ROC/PR")}\n{img("data/modeling/deep_learning_chemberta/chemberta_learning_curves.png","ChemBERTa-10M — learning curves")}\n</div>\n</section>\n\n<section id="s5"><h2>6. Stage 5 — Final Model Comparison &amp; Selection</h2>\n<pre><code>python scripts/build_final_model_comparison_table.py\npython scripts/make_research_summary_figures.py\npython scripts/make_q1_key_result_figure.py\npython scripts/make_q1_workflow_figure.py</code></pre>\n<h3>Primary seed 42 — Scaffold split (decision-relevant)</h3>\n<table>\n<tr><th class="l">Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th><th>Balanced Acc</th><th>Brier</th></tr>\n<tr><td class="l">ChemBERTa-10M</td><td>0.9511</td><td>0.9540</td><td>0.7137</td><td>0.8501</td><td>0.1132</td></tr>\n<tr class="hl"><td class="l">Molecular Graph GNN &#9733; locked</td><td>0.9780</td><td>0.9785</td><td>0.8462</td><td>0.9197</td><td>0.0620</td></tr>\n<tr><td class="l">Morgan MLP</td><td>0.9592</td><td>0.9601</td><td>0.7945</td><td>0.8951</td><td>0.0905</td></tr>\n<tr><td class="l">Morgan RF</td><td>0.9737</td><td>0.9756</td><td>0.7908</td><td>0.8919</td><td>0.0847</td></tr>\n<tr><td class="l">SMILES CNN-BiGRU</td><td>0.9645</td><td>0.9640</td><td>0.8157</td><td>0.9074</td><td>0.0758</td></tr></table>\n<h3>Primary seed 42 — Random split (reference)</h3>\n<table>\n<tr><th class="l">Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th><th>Balanced Acc</th><th>Brier</th></tr>\n<tr><td class="l">Molecular Graph GNN</td><td>0.9842</td><td>0.9850</td><td>0.8603</td><td>0.9299</td><td>0.0516</td></tr>\n<tr><td class="l">Morgan MLP</td><td>0.9894</td><td>0.9913</td><td>0.9080</td><td>0.9540</td><td>0.0332</td></tr>\n<tr><td class="l">Morgan RF</td><td>0.9909</td><td>0.9917</td><td>0.9027</td><td>0.9513</td><td>0.0560</td></tr>\n<tr><td class="l">SMILES CNN-BiGRU</td><td>0.9895</td><td>0.9898</td><td>0.8904</td><td>0.9449</td><td>0.0411</td></tr>\n<tr><td class="l">ChemBERTa-10M</td><td>0.9666</td><td>0.9673</td><td>0.8039</td><td>0.9005</td><td>0.0760</td></tr></table>\n<div class="ok"><b>Decision.</b> On the scaffold split the Molecular Graph GNN wins every primary metric (mean rank 1.00) and shows the best calibration (lowest Brier). It is selected as the locked model. Repeated-seed (N=20) scaffold: ROC-AUC 0.9776&plusmn;0.0033, MCC 0.8327&plusmn;0.0319.</div>\n<div class="grid2">\n{img("data/modeling/final_model_comparison/final_model_primary_metric_heatmap.png","Final model primary-metric heatmap")}\n{img("data/modeling/final_model_comparison/figures_research_summary/nipah_g4_q1_key_result_figure.png","Q1 key-result figure")}\n</div>\n<div class="fig">{img("data/modeling/final_model_comparison/figures_research_summary/nipah_g4_integrated_workflow.png","Integrated workflow — G4 ligand model development & candidate prioritization (updated, all stages 31 May–3 Jun 2026)")}</div>\n</section>\n\n<section id="s6"><h2>7. Stage 6 — BindingDB External Evaluation <span class="step">2–3 Jun</span></h2>\n<h3>7a. Known G4 inhibitors (positive-control)</h3>\n<pre><code>python scripts/validate_bindingdb_g4_known_inhibitors.py --cpu</code></pre>\n<ul>\n<li>BindingDB rows scanned: <b>887,473</b>; valid standardized records: 15.</li>\n<li>Seeds with evidence: <b>Phen-DC3, Telomestatin, TMPyP4</b> (3 of 8).</li>\n<li>Locked GNN: <b>3/3 scored &ge; 0.5</b>; median probability <b>0.9985</b>; all 3 with nearest-train Tanimoto &ge; 0.8.</li>\n</ul>\n<div class="note">Framing: positive-control bioactivity support (near/exact training analogs); BindingDB records are protein/helicase inhibition, not direct G4 binding constants — not independent external validation.</div>\n<div class="grid2">\n{img("data/bindingdb_g4_known_inhibitors_validation/bindingdb_g4_known_gnn_scores.png","Known G4 inhibitors — GNN scores")}\n{img("data/bindingdb_g4_known_inhibitors_validation/bindingdb_g4_known_training_similarity.png","Known G4 inhibitors — training similarity")}\n</div>\n<h3>7b. BindingDB Kd-only broad challenge &#9888;</h3>\n<pre><code>python scripts/prepare_bindingdb_kd_release.py\npython scripts/validate_bindingdb_kd_gnn_external.py</code></pre>\n<p>Release 202606 (updated 31 May); Kd rows 14,632 &rarr; dedup 7,320 &rarr; balanced sample 2,000; leakage-filtered (Tanimoto&lt;0.8) 1,997.</p>\n<table><tr><th class="l">Subset</th><th>n</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th><th>Bal.Acc</th></tr>\n<tr><td class="l">Full Kd nonambiguous</td><td>2000</td><td>0.6232</td><td>0.5918</td><td>0.0435</td><td>0.5115</td></tr>\n<tr class="hl"><td class="l">Leakage-filtered (&lt;0.8)</td><td>1997</td><td>0.6238</td><td>0.5926</td><td>0.0434</td><td>0.5115</td></tr>\n<tr><td class="l">Sensitivity (&lt;0.9)</td><td>1999</td><td>0.6237</td><td>0.5925</td><td>0.0434</td><td>0.5115</td></tr></table>\n<div class="warn">On a broad, training-distant Kd affinity space, discrimination is weak (ROC-AUC &asymp; 0.62, MCC &asymp; 0.04). This is the clearest evidence of the model\'s applicability-domain boundary and must be reported honestly.</div>\n<div class="grid2">\n{img("data/bindingdb_kd_external_validation/bindingdb_kd_primary_roc_pr_curves.png","BindingDB Kd — ROC/PR curves")}\n{img("data/bindingdb_kd_external_validation/bindingdb_kd_pkd_vs_gnn_score.png","BindingDB Kd — pKd vs GNN score")}\n</div>\n<h3>7c. Supplementary positive-control</h3>\n<pre><code>python scripts/run_bindingdb_supplementary_positive_control.py</code></pre>\n<p>7 rows / 6 unique compounds; predicted positive @0.5 = 3; near-analog &ge;0.8 = 7 rows; near-exact &ge;0.9 = 5 rows.</p>\n</section>\n\n<section id="s7"><h2>8. Stage 7 — PubChem External Evaluation <span class="step">3 Jun</span></h2>\n<h3>8a. BioAssay strict checkpoint (two-step)</h3>\n<pre><code>python scripts/collect_pubchem_g4_bioassay_aids.py            # AID checkpoint\npython scripts/collect_pubchem_g4_compound_level_records.py  # compound-level + QC</code></pre>\n<ul>\n<li>Selected AIDs: <b>282</b> (confirmed 168, probable 114); raw records 1,612 &rarr; dedup 417.</li>\n<li>After conservative labeling + exact training-overlap removal: <b>48 compounds (42 active : 6 inactive)</b> — small and imbalanced.</li>\n</ul>\n<div class="grid2">\n{img("data/pubchem_g4_bioassay_external_validation/pubchem_g4_final_label_counts.png","PubChem strict — label counts")}\n{img("data/pubchem_g4_bioassay_external_validation/pubchem_g4_final_similarity_distribution.png","PubChem strict — similarity distribution")}\n</div>\n<h3>8b. Expanded evidence + background</h3>\n<pre><code>python scripts/collect_pubchem_g4_expanded_external_evidence_set.py\npython scripts/validate_pubchem_g4_expanded_locked_gnn.py</code></pre>\n<p>All-evidence nonredundant 773; strict no-overlap <b>442</b>. Tiers: T1 actives 281 (overlap 239), T1 confirmed inactive 7, inactive-like 20, unresolved 109, T2 annotation 6, T3 background/decoy 350.</p>\n<div class="grid2">\n{img("data/pubchem_g4_expanded_external_evidence_set/pubchem_g4_expanded_tier_counts.png","Expanded — tier counts")}\n{img("data/pubchem_g4_expanded_external_evidence_set/pubchem_g4_expanded_training_overlap_counts.png","Expanded — training overlap")}\n{img("data/pubchem_g4_expanded_external_evidence_set/pubchem_g4_expanded_similarity_distribution.png","Expanded — similarity distribution")}\n{img("data/pubchem_g4_expanded_external_evidence_set/pubchem_g4_expanded_property_distributions.png","Expanded — property distributions")}\n</div>\n</section>\n\n<section id="s8"><h2>9. Stage 8 — Imbalance-aware Analysis &amp; Robustness</h2>\n<pre><code>python scripts/analyze_pubchem_g4_locked_gnn_imbalance.py</code></pre>\n<p>Merged benchmark <b>n=402</b> (positives 46 / negatives 356).</p>\n<div class="kpi">\n<div><span>PR-AUC</span><b>0.8319</b></div><div><span>ROC-AUC</span><b>0.9759</b></div>\n<div><span>MCC</span><b>0.7642</b></div><div><span>Balanced Acc</span><b>0.9218</b></div>\n<div><span>Sensitivity</span><b>0.8913</b></div><div><span>Specificity</span><b>0.9522</b></div>\n</div>\n<p>Confusion matrix: TN 339, FP 17, FN 5, TP 41.</p>\n<h3>Top-k enrichment</h3>\n<table><tr><th>Top %</th><th>k</th><th>positives recovered</th><th>precision@k</th><th>enrichment</th></tr>\n<tr class="hl"><td>1%</td><td>5</td><td>5/46</td><td>1.000</td><td>8.74&times;</td></tr>\n<tr><td>5%</td><td>21</td><td>18/46</td><td>0.857</td><td>7.49&times;</td></tr>\n<tr><td>10%</td><td>41</td><td>34/46</td><td>0.829</td><td>7.25&times;</td></tr></table>\n<h3>Balanced negative resampling (1,000 iterations, 46 vs 46)</h3>\n<table><tr><th class="l">Metric</th><th>mean</th><th>95% interval</th></tr>\n<tr><td class="l">ROC-AUC</td><td>0.9759</td><td>0.9485 &ndash; 0.9934</td></tr>\n<tr><td class="l">PR-AUC</td><td>0.9710</td><td>0.9232 &ndash; 0.9939</td></tr>\n<tr><td class="l">MCC</td><td>0.8465</td><td>0.7826 &ndash; 0.8966</td></tr>\n<tr><td class="l">Balanced Acc</td><td>0.9222</td><td>0.8913 &ndash; 0.9457</td></tr>\n<tr><td class="l">Specificity</td><td>0.9532</td><td>0.8913 &ndash; 1.0000</td></tr></table>\n<div class="grid2">\n{img("data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/imbalance_sensitivity/merged_pubchem_locked_gnn_roc_pr_curves.png","Merged benchmark — ROC/PR")}\n{img("data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/imbalance_sensitivity/merged_pubchem_locked_gnn_topk_enrichment.png","Merged benchmark — top-k enrichment")}\n{img("data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/imbalance_sensitivity/merged_pubchem_locked_gnn_score_distribution_by_label.png","Score distribution by evidence label")}\n{img("data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/imbalance_sensitivity/balanced_negative_resampling_metric_distributions.png","Balanced resampling — metric distributions")}\n</div>\n<div class="note">Tier-1-only subset (n=48) has only 6 negatives (ROC-AUC 0.4762, MCC 0.1627) and must not be used as a standalone strong external-validation claim.</div>\n</section>\n\n<section id="repro"><h2>10. Reproduce End-to-End (Command Sequence)</h2>\n<pre><code># 0. Environment\nconda env create -f environment.yml\npython scripts/check_environment.py\n\n# 1. Dataset\npython scripts/fetch_g4ldb.py\npython scripts/preprocess_g4ldb_ml_ready.py\npython scripts/prepare_chembl_negative_decoys.py\npython scripts/visualize_binary_dataset.py\npython scripts/plot_physicochemical_boxplots.py\n\n# 2. Train models\npython scripts/train_baseline_g4_classifier.py\npython scripts/train_mlp_morgan_g4_classifier.py\npython scripts/train_smiles_sequence_g4_classifier.py\npython scripts/train_chemberta_g4_classifier.py\npython scripts/train_gnn_molecular_graph_g4_classifier.py\n\n# 3. Validate + compare\npython scripts/validate_baseline_g4_classifier.py\npython scripts/build_final_model_comparison_table.py\npython scripts/make_research_summary_figures.py\n\n# 4. BindingDB external\npython scripts/validate_bindingdb_g4_known_inhibitors.py --cpu\npython scripts/prepare_bindingdb_kd_release.py\npython scripts/validate_bindingdb_kd_gnn_external.py\npython scripts/run_bindingdb_supplementary_positive_control.py\n\n# 5. PubChem external\npython scripts/collect_pubchem_g4_bioassay_aids.py\npython scripts/collect_pubchem_g4_compound_level_records.py\npython scripts/collect_pubchem_g4_expanded_external_evidence_set.py\npython scripts/validate_pubchem_g4_expanded_locked_gnn.py\n\n# 6. Imbalance / robustness\npython scripts/analyze_pubchem_g4_locked_gnn_imbalance.py\n\n# 7. Reports\npython scripts/build_master_report_html.py\npython scripts/build_reproducibility_workflow_html.py</code></pre>\n</section>\n\n<section id="limit"><h2>11. Limitations</h2>\n<ul>\n<li><b>Training overlap:</b> most G4 actives are exact training analogs (median nearest-train similarity 1.0 for Tier-1 actives); strict no-overlap shrinks usable data to 48 in the strict PubChem checkpoint.</li>\n<li><b>Scarce confirmed inactives:</b> only ~6–7 experimentally confirmed inactive/inactive-like compounds.</li>\n<li><b>Decoy-based negatives:</b> 350/356 negatives in the merged benchmark are property-shaped putative decoys.</li>\n<li><b>Small positive count:</b> only 46 evidence-supported positives in the merged benchmark.</li>\n<li><b>Limited broad generalization:</b> BindingDB Kd-only ROC-AUC &asymp; 0.62.</li>\n</ul>\n</section>\n\n<section id="concl"><h2>12. Manuscript-safe Conclusions</h2>\n<div class="quote">Using a strict no-training-overlap PubChem-derived benchmark, the locked Molecular Graph GNN demonstrated strong enrichment and prioritization of G4 evidence-supported compounds against property-shaped PubChem background comparators (PR-AUC 0.83, ROC-AUC 0.98, top-1% enrichment 8.7&times;). Balanced negative subsampling (1,000 iterations) confirmed robustness to class imbalance (ROC-AUC 0.976, 95% CI 0.949–0.993). Because most negatives were putative decoys rather than experimentally confirmed inactives, these results are interpreted as external enrichment/prioritization performance rather than a fully experimental active-versus-inactive validation. On a broad BindingDB Kd-only affinity challenge distant from the training domain, discrimination was limited (ROC-AUC 0.62), delineating the model\'s applicability domain. Known G4 inhibitors with BindingDB records were recovered as positive-control bioactivity support.</div>\n<p><b>Companion artifacts:</b> <code>MASTER_REPORT_NIPAH_G4_2026-05-31_to_2026-06-03.md</code> / <code>.html</code>.</p>\n</section>\n\n<footer>Full reproducibility workflow generated {GEN} · Luke for Luktei · figures embedded inline (base64) · numbers verified against project files</footer>\n</main></body></html>\n"""\n\nwith open(OUT, "w") as f:\n    f.write(html)\nsize = os.path.getsize(OUT)\nprint(f"WROTE {OUT}")\nprint(f"SIZE {size} bytes ({size/1024/1024:.2f} MB)")\n'


import argparse
import contextlib
import io
import json
import os
import shlex
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO


class Tee(io.TextIOBase):
    """Write text both to the live terminal stream and to a log file."""

    def __init__(self, live: TextIO, log: TextIO):
        self.live = live
        self.log = log

    def write(self, s: str) -> int:
        self.live.write(s)
        self.live.flush()
        self.log.write(s)
        self.log.flush()
        return len(s)

    def flush(self) -> None:
        self.live.flush()
        self.log.flush()


def parse_seed_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in str(value).replace(',', ' ').split():
            part = part.strip()
            if not part:
                continue
            if '-' in part and part.replace('-', '').isdigit():
                a, b = part.split('-', 1)
                start, end = int(a), int(b)
                if end < start:
                    raise SystemExit(f"Invalid seed range: {part}")
                out.extend(str(i) for i in range(start, end + 1))
            else:
                int(part)
                out.append(part)
    return out


def build_forwarded_args(args: argparse.Namespace, unknown_args: list[str], seed_mode: str) -> tuple[list[str], list[str]]:
    forwarded: list[str] = []
    notes: list[str] = []

    if seed_mode == 'single':
        if args.seed is not None:
            forwarded += ['--seed', str(args.seed)]
            notes.append(f'seed={args.seed}')
    elif seed_mode == 'repeated':
        primary = args.primary_seed if args.primary_seed is not None else args.seed
        if primary is not None:
            forwarded += ['--primary-seed', str(primary)]
            notes.append(f'primary_seed={primary}')
        seeds = parse_seed_list(args.seeds)
        if args.n_seeds is not None:
            if args.n_seeds < 1:
                raise SystemExit('--n-seeds must be >= 1')
            seeds = [str(i) for i in range(args.n_seeds)]
        if seeds:
            forwarded += ['--seeds', *seeds]
            notes.append(f'repeated_seeds={" ".join(seeds)}')
    elif seed_mode in {'none', 'env'}:
        if args.seed is not None or args.primary_seed is not None or args.seeds is not None or args.n_seeds is not None:
            notes.append('Seed options were supplied but this embedded step has no seed argument; seed options were logged but not passed.')
    else:
        notes.append(f'Unknown seed_mode={seed_mode!r}; seed options were not passed automatically.')

    if args.extra_args.strip():
        extra = shlex.split(args.extra_args)
        forwarded += extra
        notes.append(f'extra_args={args.extra_args}')

    if unknown_args:
        forwarded += unknown_args
        notes.append(f'unknown_args_forwarded={" ".join(shlex.quote(x) for x in unknown_args)}')

    return forwarded, notes


def start_progress_thread(step_id: str, step_title: str, interval: float, stop_event: threading.Event) -> threading.Thread | None:
    if interval <= 0:
        return None

    def loop() -> None:
        started = time.monotonic()
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - started)
            print(f"[RUNNING] {datetime.now().isoformat(timespec='seconds')} | {step_id} {step_title} | elapsed={elapsed}s", flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def run_embedded_step(
    *,
    step_id: str,
    step_title: str,
    embedded_source_name: str,
    seed_mode: str,
    embedded_source: str,
) -> int:
    parser = argparse.ArgumentParser(description=f"Standalone v.4 reproducibility step: {step_title}")
    parser.add_argument('--project-root', default=None, help='Output/project root. Default: this script folder. No original-script folder is required.')
    parser.add_argument('--log-dir', default=None, help='Directory for step logs. Default: <workflow folder>/logs.')
    parser.add_argument('--seed', type=int, default=None, help='Single seed for steps that support --seed. Also used as --primary-seed for repeated-seed steps unless --primary-seed is set.')
    parser.add_argument('--primary-seed', type=int, default=None, help='Primary seed for repeated-seed scripts that support --primary-seed.')
    parser.add_argument('--seeds', nargs='*', default=None, help='Repeated split seeds for scripts that support --seeds. Accepts space/comma lists or ranges, e.g. --seeds 0-19 or --seeds 0 1 2.')
    parser.add_argument('--n-seeds', type=int, default=None, help='Shortcut for --seeds 0-(N-1) on repeated-seed scripts.')
    parser.add_argument('--extra-args', default='', help='Additional arguments passed verbatim to the embedded source, e.g. "--cpu --batch-size 64".')
    parser.add_argument('--progress-interval', type=float, default=30.0, help='Seconds between [RUNNING] heartbeat messages during long steps. Use 0 to disable.')
    parser.add_argument('--dry-run', action='store_true', help='Print/log what would run without executing the embedded source.')
    args, unknown_args = parser.parse_known_args()

    wrapper_path = Path(__file__).resolve()
    workflow_dir = wrapper_path.parent
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else workflow_dir
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else workflow_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    forwarded_args, notes = build_forwarded_args(args, unknown_args, seed_mode)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = f'{step_id}_{Path(embedded_source_name).stem}_{timestamp}'
    stdout_path = log_dir / f'{base}.stdout.log'
    stderr_path = log_dir / f'{base}.stderr.log'
    status_json = log_dir / f'{base}.status.json'
    status_md = log_dir / f'{base}.status.md'
    virtual_source_path = project_root / '_embedded_source_virtual' / embedded_source_name

    status = {
        'version': 'revised_reproduce_workflow_v.4',
        'step_id': step_id,
        'step_title': step_title,
        'embedded_source': embedded_source_name,
        'workflow_dir': str(workflow_dir),
        'project_root': str(project_root),
        'argv_for_embedded_source': [str(virtual_source_path), *forwarded_args],
        'started': datetime.now().isoformat(timespec='seconds'),
        'seed_mode': seed_mode,
        'notes': notes,
        'dry_run': bool(args.dry_run),
        'stdout_log': str(stdout_path),
        'stderr_log': str(stderr_path),
        'status_json': str(status_json),
        'status_md': str(status_md),
    }

    returncode: int | None = None
    result = 'UNKNOWN'
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    stop_event = threading.Event()
    progress_thread: threading.Thread | None = None

    with stdout_path.open('w', encoding='utf-8') as stdout_log, stderr_path.open('w', encoding='utf-8') as stderr_log:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_log)), contextlib.redirect_stderr(Tee(sys.__stderr__, stderr_log)):
            print(f"[START] {status['started']} | {step_id} {step_title}", flush=True)
            print(f"Workflow dir: {workflow_dir}", flush=True)
            print(f"Project root: {project_root}", flush=True)
            print(f"Embedded source: {embedded_source_name}", flush=True)
            print('Forwarded args:', ' '.join(shlex.quote(x) for x in forwarded_args) if forwarded_args else '(none)', flush=True)
            print(f"Logs: {log_dir}", flush=True)
            print(f"Stdout log: {stdout_path}", flush=True)
            print(f"Stderr log: {stderr_path}", flush=True)

            if args.dry_run:
                result = 'DRY_RUN'
                returncode = None
                print('[RUNNING] dry-run requested; embedded source not executed.', flush=True)
            else:
                print(f"[RUNNING] executing embedded source now; live output is mirrored to logs every print/flush.", flush=True)
                try:
                    project_root.mkdir(parents=True, exist_ok=True)
                    os.chdir(project_root)
                    sys.argv = [str(virtual_source_path), *forwarded_args]
                    progress_thread = start_progress_thread(step_id, step_title, args.progress_interval, stop_event)
                    namespace = {
                        '__name__': '__main__',
                        '__file__': str(virtual_source_path),
                        '__package__': None,
                        '__builtins__': __builtins__,
                    }
                    exec(compile(embedded_source, str(virtual_source_path), 'exec'), namespace)
                    returncode = 0
                except SystemExit as exc:
                    code = exc.code
                    if code is None:
                        returncode = 0
                    elif isinstance(code, int):
                        returncode = code
                    else:
                        print(code, file=sys.stderr)
                        returncode = 1
                except BaseException:
                    traceback.print_exc()
                    returncode = 1
                finally:
                    stop_event.set()
                    if progress_thread is not None:
                        progress_thread.join(timeout=1)
                    sys.argv = old_argv
                    os.chdir(old_cwd)

                result = 'PASS' if returncode == 0 else 'FAIL'

            finished = datetime.now().isoformat(timespec='seconds')
            status.update({'finished': finished, 'returncode': returncode, 'result': result})
            print(f"[DONE] {finished} | {step_id} {step_title} | result={result} | returncode={returncode}", flush=True)
            print(f"Status: {status_md}", flush=True)

    status_json.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    cmd_text = ' '.join(shlex.quote(x) for x in [Path(sys.executable).name, wrapper_path.name, *sys.argv[1:]])
    status_md.write_text(
        f"# {step_id} {step_title}\n\n"
        f"- Version: `revised_reproduce_workflow_v.4`\n"
        f"- Started: {status['started']}\n"
        f"- Finished: {status.get('finished')}\n"
        f"- Result: **{result}**\n"
        f"- Return code: `{returncode}`\n"
        f"- Embedded source: `{embedded_source_name}`\n"
        f"- Workflow dir: `{workflow_dir}`\n"
        f"- Project root: `{project_root}`\n"
        f"- Seed mode: `{seed_mode}`\n"
        f"- Notes: {', '.join(notes) if notes else 'none'}\n"
        f"- Wrapper command observed: `{cmd_text}`\n"
        f"- Forwarded args: `{(' '.join(shlex.quote(x) for x in forwarded_args) if forwarded_args else '(none)')}`\n"
        f"- Stdout log: `{stdout_path}`\n"
        f"- Stderr log: `{stderr_path}`\n"
        f"- Status JSON: `{status_json}`\n",
        encoding='utf-8',
    )

    if args.dry_run:
        return 0
    return int(returncode or 0)



if __name__ == '__main__':
    raise SystemExit(run_embedded_step(
        step_id=STEP_ID,
        step_title=STEP_TITLE,
        embedded_source_name=EMBEDDED_SOURCE_NAME,
        seed_mode=SEED_MODE,
        embedded_source=EMBEDDED_SOURCE,
    ))
