# Implementation Map

Every numbered equation, algorithm box, reported table, ablation row, figure, and metric in the paper is mapped to a file path under `cartidt/` and to a test under `tests/`. Rows are grouped by manuscript section.

Module slug legend: `IN` = frontend, `BK` = passes/encode, `DC` = passes/segment, `TW` = passes/twin, `EV` = passes/evidence, `TR` = backend, `EL` = diagnostics, `EN` = driver.

## Section III — Methodology

| Paper anchor                | Item                                                  | File path                                              | Module | Test                                                       | Notes |
| --------------------------- | ----------------------------------------------------- | ------------------------------------------------------ | ------ | ---------------------------------------------------------- | ----- |
| §III.A                      | Problem formulation (X, Y, ŷ, u_ale, u_epi)            | `cartidt/frontend/schema.py`                     | IN     | `tests/shape/test_contracts.py`                            | typed dataclasses for batch / prediction tuples |
| §III.B                      | Three-module overview                                 | `cartidt/link.py` (`CartiDT` + `build_model`)      | —      | —                                                          | thin facade composing BK / DC / TW / EV |
| §III.C-1, Eq (1)            | LoRA injection on Q, V (r = 16, α = 32)                | `cartidt/passes/encode/lora.py`                | BK     | `tests/regression/test_lora_freeze_signature.py`           | adapted weight = W_frozen + (α / r) · B A; rank-16 |
| §III.C-2, Eq (2)            | Depth-axis positional interpolation                   | `cartidt/passes/encode/depth_posembed.py`                  | BK     | `tests/shape/test_pos_depth_shape.py`                      | bicubic on P_2D + learnable P_depth, stride s_d = 4 |
| §III.C-3                    | UPerNet decoder (FPN + PPM)                           | `cartidt/passes/segment/upernet.py`                     | DC     | `tests/shape/test_upernet_shape.py`                        | full-resolution 7-class logits |
| §III.C-4                    | Joint training of seg ↔ grading                       | `cartidt/backend/objective.py`               | TR     | `tests/integration/test_four_step_pipeline.py`             | gradient signals from EDL reach decoder |
| §III.D-1                    | Motivation (compartment coupling)                     | `cartidt/passes/twin/anatomy_graph.py` (docstring + edge list) | TW   | `tests/shape/test_topology.py`                             | 14 nodes / 26 directed edges |
| §III.D-2                    | Graph construction (node list)                        | `cartidt/passes/twin/anatomy_graph.py`                       | TW     | `tests/shape/test_topology.py`                             | enumerated node table |
| §III.D-2 final ¶            | Edge weight estimation (frozen, contact-area ratio)   | `cartidt/passes/twin/contact_weights.py`                   | TW     | `tests/regression/test_edge_weights_fixed.py`              | computed once from mean training seg |
| §III.D-3, Eq (3)            | Node feature extraction (mean over masked features)   | `cartidt/passes/twin/region_features.py`                  | TW     | `tests/shape/test_node_features_shape.py`                  | per-subject masked GAP |
| §III.D-4, Eq (4)            | GraphSAGE-style message passing (L = 3, d_g = 256)     | `cartidt/passes/twin/graph_sage.py`                           | TW     | `tests/shape/test_sage_shape.py`                           | hand-rolled, edge weights as scaling |
| §III.E-1, Eq (5)            | Dirichlet PDF                                         | `cartidt/passes/evidence/dirichlet.py`                  | EV     | `tests/regression/test_dirichlet_pdf_vs_scipy.py`          | numeric check vs `scipy.stats.dirichlet` |
| §III.E-1, Eq (6)            | Concentration parameters from MLP via Softplus + 1    | `cartidt/passes/evidence/dirichlet_head.py`                       | EV     | `tests/shape/test_evidence_head_shape.py`                  | 2-layer MLP, K = 4 |
| §III.E-1, Eq (7)            | Aleatoric u_ale = (K / α₀(α₀+1)) · Σ π̂(1−π̂)            | `cartidt/passes/evidence/decompose.py`                | EV     | `tests/regression/test_uncertainty_closed_form.py`         | analytic check, K = 4 |
| §III.E-1, Eq (8)            | Epistemic u_epi = K / α₀                              | `cartidt/passes/evidence/decompose.py`                | EV     | `tests/regression/test_uncertainty_closed_form.py`         | same file |
| §III.E-2, Eq (9)            | EDL loss = digamma term + λ_t KL(Dir(α̃)‖Dir(1))       | `cartidt/passes/evidence/edl.py`                   | EV     | `tests/regression/test_edl_loss_vs_sensoy.py`              | T_anneal = 10 epochs |
| §III.F, Eq (10)             | Joint loss L = L_seg + μ L_EDL, μ = 0.5               | `cartidt/backend/objective.py`               | TR     | `tests/integration/test_four_step_pipeline.py`             | balances Dice + CE + EDL |
| §III.F                      | L_seg = soft-Dice + CE                                | `cartidt/backend/segmentation_loss.py`                    | TR     | `tests/regression/test_soft_dice_vs_monai.py`              | numeric agreement with monai |
| §III.G                      | Complexity analysis                                   | `cartidt/diagnostics/cost.py`                     | EL     | `tests/shape/test_flop_count.py`                           | param + FLOP counters used by Table X |

## Section IV — Experiments

| Paper anchor      | Item                                                      | File path                                       | Module | Test                                             | Notes |
| ----------------- | --------------------------------------------------------- | ----------------------------------------------- | ------ | ------------------------------------------------ | ----- |
| §IV.A, Table II   | Dataset catalogue (OAI, OAI-ZIB, IWOAI, SKI10)            | `cartidt/frontend/catalog.py`               | IN     | `tests/shape/test_registry.py`                   | dataset factory + split metadata |
| §IV.A-1           | OAI 3D DESS loader, WORMS grading labels                  | `cartidt/frontend/sources/oai_dess.py`                    | IN     | `tests/shape/test_oai_loader.py`                 | reads 507 ZIB segs + 3212 grading subs |
| §IV.A-1           | OAI grading-split logic (3212 / 802 / 782)                | `cartidt/frontend/sources/oai_dess.py`                    | IN     | `tests/shape/test_oai_loader.py`                 | exact split sizes |
| §IV.A-2           | SKI10 loader (femoral + tibial only)                      | `cartidt/frontend/sources/ski10.py`                  | IN     | `tests/shape/test_ski10_loader.py`               | zero-shot transfer target |
| §IV.A-3           | IWOAI partition (60-14-14 ×2 = 176 vols)                  | `cartidt/frontend/sources/iwoai_split.py`                  | IN     | `tests/shape/test_iwoai_loader.py`               | uses official IWOAI test split |
| §IV.B             | Baseline registry (3D V-Net, Att-UNet, TransUNet, MedSAM, SAM-Med3D, SAMRI-2, Prob U-Net) | `cartidt/diagnostics/reference_models.py` | EL | `tests/shape/test_baseline_registry.py`     | descriptive; we only score them |

## Section V — Implementation, Metrics, Results, Ablations

| Paper anchor       | Item                                                         | File path                                       | Module | Test                                                       | Notes |
| ------------------ | ------------------------------------------------------------ | ----------------------------------------------- | ------ | ---------------------------------------------------------- | ----- |
| §V.C               | AdamW, cosine + warmup, batch 4 × accum 2 × 4 GPUs, 120 ep   | `cartidt/backend/optimizer.py` + `cartidt/backend/schedule.py` | TR | `tests/regression/test_scheduler_curve.py`                 | numeric LR-curve check at epoch 0, 5, 60, 120 |
| §V.C               | Augmentations (rotation / scale / translate / elastic / noise / gain) | `cartidt/frontend/conditioning/spatial_aug.py` | IN | `tests/shape/test_augment_shape.py`                        | each transform produces same shape and dtype |
| §V.C               | Seeds 42, 123, 7                                             | `cartidt/backend/reproducibility.py`            | TR     | `tests/regression/test_determinism.py`                     | deterministic loss on fixed batch |
| §V.D, Table III    | DSC, ASSD per-compartment                                    | `cartidt/diagnostics/overlap.py`             | EL     | `tests/regression/test_dsc_vs_monai.py`                    | DSC ↑, ASSD ↓ (mm) |
| §V.D, Table IV     | AUROC, κ_w (quadratic weighted)                              | `cartidt/diagnostics/ordinal.py`         | EL     | `tests/regression/test_auroc_vs_sklearn.py`                | binary "any-damage" AUROC; κ_w via sklearn |
| §V.D, Table V      | ECE, AURC, FPR@τ                                             | `cartidt/diagnostics/calibration.py`           | EL     | `tests/regression/test_ece_curve.py`                       | τ = 0.15 by default; configurable |
| §VI ¶1             | Paired bootstrap n = 10 000, Holm–Bonferroni                 | `cartidt/diagnostics/significance.py`              | EL     | `tests/regression/test_bootstrap_known_input.py`           | CI matches analytic for Gaussian sample |
| §V.E-1, Table VI   | Transfer ablation (6 rows)                                   | `configs/ablation_transfer_*.yaml`              | configs | `tests/integration/test_config_inflation.py`              | switch governs LoRA r, depth interp, full-FT |
| §V.E-2, Table VII  | GNN ablation (5 rows)                                        | `configs/ablation_gnn_*.yaml`                   | configs | `tests/integration/test_config_inflation.py`              | switch governs node-aggregator |
| §V.E-3, Table VIII | Component ablation (7 rows)                                  | `configs/ablation_component_*.yaml`             | configs | `tests/integration/test_config_inflation.py`              | switch governs component removal |
| §V.E, Table IX     | Zero-shot SKI10 transfer                                     | `configs/transfer_ski10.yaml` + `cartidt/driver/evaluate.py` | configs/EN | `tests/integration/test_ski10_transfer_glue.py`         | inference-only on SKI10 |
| §V.E, Table X      | Compute cost                                                 | `cartidt/diagnostics/cost.py`              | EL     | `tests/shape/test_flop_count.py`                           | params + FLOPs + per-volume timing |

## Figures

| Figure | Item                                                        | File path                                     | Module | Notes |
| ------ | ----------------------------------------------------------- | --------------------------------------------- | ------ | ----- |
| Fig 1  | Architecture diagram                                        | `assets/fig1_architecture.svg`                | —      | static SVG transcribed from the manuscript layout — no code path emits it |
| Fig 2  | Mathematical-principle diagram                              | `assets/fig2_principle.svg`                   | —      | static SVG; conceptual figure with no code path |
| Fig 3  | Qualitative seg + uncertainty maps                          | `cartidt/driver/panel_uncertainty.py`   | EN     | renders panels (b)–(j) from a trained ckpt |
| Fig 4  | Chord + boxplot + Sankey + heatmap analyses                 | `cartidt/driver/panel_analysis.py`      | EN     | reads metrics csv emitted by evaluate |

## Coverage check

- Numbered equations: (1)–(10) all mapped.
- Tables: II–X all mapped (Table I is positioning prose only).
- Figures: 1–4 all mapped (Figs 1–2 are static).
- Ablations: 6 + 5 + 7 = 18 ablation rows + Table IX + Table X all mapped.
- Metrics: DSC, ASSD, AUROC, κ_w, ECE, AURC, FPR@τ all mapped.

No paper item is unmatched.
