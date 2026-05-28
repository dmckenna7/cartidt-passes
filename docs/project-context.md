# Project Context

This document records the values that every other turn treats as canonical. All anchors point to the manuscript; confidence is HIGH unless flagged.

## Executive summary

| Field               | Value                                                            | Confidence |
| ------------------- | ---------------------------------------------------------------- | ---------- |
| project_name        | `cartidt`                                                        | HIGH       |
| domain              | musculoskeletal MRI — cross-modal FM transfer for 3D knee cartilage segmentation, WORMS damage grading, and decomposed evidential uncertainty | HIGH |
| framework           | PyTorch 2.1.x                                                    | HIGH       |
| venue               | IEEE Journal of Biomedical and Health Informatics (J-BHI)        | HIGH       |
| primary_datasets    | 3 public corpora (see §6)                                        | HIGH       |
| compute_target      | 4× NVIDIA A100 80GB, ~38 h wall-clock                            | HIGH       |
| hparams_reference   | Methods §V.C "Implementation Details" + Sec IV.A datasets        | HIGH       |
| supp_path           | none — no supplementary file accompanies the submission          | HIGH       |
| extra_signals       | three algorithmic boxes (Modules A/B/C), code-availability statement present, ethics statement present | HIGH |

NEEDS_USER_DECISION: 1 (training precision) — see §7.

---

## 1. project_name

`cartidt` — snake_case Python package name derived from the system's own acronym **CartiDT** (Cartilage Digital Twin) introduced in the abstract and Sec II final paragraph.

## 2. supp_path

None. The submission is a single 12-page PDF (running title "FENG et al.: FOUNDATION MODELS MEET MUSCULOSKELETAL DIGITAL TWINS"). No `*supp*`, `*SI*`, or appendix sibling files were located.

## 3. domain

Computational musculoskeletal imaging. The system targets early (preclinical / WORMS KL 0–1) cartilage degeneration on 3D double-echo steady-state (DESS) knee MRI. The work bundles three subproblems:

- six-compartment dense cartilage + meniscus segmentation (Femoral, Medial Tibial, Lateral Tibial, Patellar, Medial Meniscus, Lateral Meniscus),
- per-compartment WORMS damage grading on K = 4 ordinal grades (0 / 1 / 2 / 3),
- per-prediction decomposition of predictive uncertainty into aleatoric `u_ale` and epistemic `u_epi` components used for selective abstention.

## 4. framework

PyTorch 2.1.x. Anchor: Sec V.C — *"CartiDT has been designed for use with PyTorch 2.1"*. Backbone weights are pulled from the DINOv2 release (Sec III.C, ref. [13]); the GNN is GraphSAGE-style (ref. [15]); the evidential head follows Sensoy et al. (ref. [10]). Implementation choices that follow from these citations:

- `torch>=2.1,<2.2`, `torchvision`, `timm` for ViT-L weight loading (or `transformers` `facebook/dinov2-large`),
- `einops` for the depth-axis token reshape,
- a hand-rolled GraphSAGE layer (faster than pulling `torch-geometric` for a 14-node, 26-edge graph),
- `monai` for soft-Dice and random elastic deformation,
- `scipy` for paired bootstrap, `statsmodels` for Holm–Bonferroni.

## 5. venue

IEEE Journal of Biomedical and Health Informatics. Anchor: running header "IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS" on every page; two-column IEEE template; numbered references (Vancouver). Sec V.C confirms the journal scope (clinical / biomedical informatics).

## 6. primary_datasets

| Name                            | Version           | Access                                  | License / DUA                                       |
| ------------------------------- | ----------------- | --------------------------------------- | --------------------------------------------------- |
| Osteoarthritis Initiative (OAI) | 4796-subject full release; this work uses 507 baseline 3D DESS volumes for segmentation and 4796 subjects (3212 / 802 / 782 train / val / test) for WORMS grading | <https://nda.nih.gov/oai> | NIH Data Use Agreement (DUA protocol 2024-0347 per the manuscript's Ethics Statement); restricted — registration required |
| OAI-ZIB (ZIB segmentations of OAI) | 507 manually segmented volumes, strict subject-disjoint subset of OAI | distributed by the Zuse Institute Berlin alongside OAI | Same OAI DUA + ZIB redistribution terms; restricted |
| SKI10                           | MICCAI 2010 challenge release — 100 training / 50 test knees with bone + cartilage masks for femur and tibia (two compartments) | <http://www.ski10.org> | Public, challenge-distribution terms |
| IWOAI 2019 partition (derived)  | 88-subject subset of OAI-ZIB at two time-points → 176 volumes; 60-14-14 train/val/test split | <https://github.com/IWOAI/knee-segmentation> | Inherits OAI/OAI-ZIB terms |

All four are publicly available subject to their respective DUAs. The manuscript trains its segmentation losses on the 404 ZIB volumes that are not in the 88-subject IWOAI val/test partition and trains its grading loss on the 3212 OAI grading-train subjects (Sec IV.A ¶3).

## 7. compute_target

- Hardware: 4× NVIDIA A100 80GB (Sec V.C ¶1).
- Wall-clock: ~38 h to full convergence at 120 epochs (Sec V.C ¶1).
- Storage: ZIB 507 vols × 160 × 384 × 384 × 2 bytes ≈ 24 GB raw; OAI grading metadata ≈ negligible. Reserve ≈ 80 GB for processed shards + checkpoints.
- Effective batch size: per-GPU batch 4 × grad-accum 2 × 4 GPUs = **32** (used as the divisor for the LR schedule).
- Precision: **NEEDS_USER_DECISION**. The manuscript states `weight_decay=1e-2` and does not state fp16 / bf16 / tf32 anywhere in Methods or Implementation Details. A100 + LoRA + frozen ViT-L typically runs in bf16. Candidates, ranked:
  1. **bf16-autocast** (recommended). A100 supports it natively; no loss-scaling needed; matches the typical DINOv2 fine-tuning recipe.
  2. fp16-autocast with `torch.cuda.amp.GradScaler` — slightly faster than bf16 on A100, but susceptible to under/overflow in the EDL KL term.
  3. tf32-only — safest but ≈ 1.7× slower; would not fit a 38-h wall-clock on 4× A100.

User to confirm before Turn 1 lands.

## 8. hparams_reference

Hyperparameters are stated in prose in §V.C ("Implementation Details"), with augmentation details in the same paragraph and dataset splits in §IV.A. The full canonical set:

| Group              | Value                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------- |
| backbone           | DINOv2 ViT-Large/16, 304 M params, **frozen** (Sec III.C)                              |
| LoRA               | r = 16, α = 32, on the Q and V projections of all 24 self-attention layers (Eq 1, Sec III.C); 1.57 M trainable |
| depth positional   | bicubic interpolation of `P_2D` over (H/u)² grid + learned `P_depth ∈ R^{D/s_d × d}`, `s_d = 4` (Eq 2) |
| decoder            | UPerNet-style (FPN + PPM) at full resolution, 23.4 M (Sec III.C-3)                     |
| GNN                | GraphSAGE L = 3, hidden d_g = 256, GELU; 14 nodes / 26 directed edges (Sec III.D, Eq 4) |
| GNN edge weights   | contact-area ratio of thickening-normalised surface overlap at the seg boundary; computed once from the mean training segmentation and frozen for both train and test (Sec III.D-2) |
| evidence head      | 2-layer MLP → Softplus(·) + 1 → Dirichlet α; K = 4 (Eq 5–6); 0.3 M                       |
| evidence loss      | E[(y − π)²] term + λ_t · KL(Dir(α̃)‖Dir(1)), λ_t = min(1, t / T_anneal), T_anneal = 10 epochs (Eq 9) |
| joint loss         | L = L_seg + μ · L_EDL, μ = 0.5; L_seg = soft-Dice + cross-entropy (Eq 10)               |
| optimiser          | AdamW (β1 = 0.9, β2 = 0.999, wd = 1e-2)                                                 |
| LR schedule        | linear warmup 0 → 2e-4 over the first 5 epochs, then cosine decay to a floor of 1e-6 over remaining 115 epochs |
| batch              | 4 per GPU × 2 grad-accum steps × 4 GPUs = effective 32                                 |
| epochs             | 120                                                                                   |
| seeds              | 42, 123, 7 (three independent runs; ±std reported in all main tables)                  |
| volume shape       | 160 × 384 × 384, voxel spacing 0.365 × 0.365 × 0.7 mm, z-score normalised per volume   |
| augmentations      | rotation ±10°, scale 0.9–1.1, translation ±15 voxels, random elastic, additive Gaussian σ = 0.02, multiplicative gain in [0.95, 1.05] |
| inference          | single forward pass (no test-time augmentation in main results)                        |
| stats              | paired bootstrap n = 10 000; Holm–Bonferroni correction across Tables III–IX (ref. [44]) |

## 9. extra_signals

- **Algorithm boxes**: three — Module A (LoRA + depth-axis interpolation, Eq 1–2), Module B (graph aggregation, Eq 3–4), Module C (Dirichlet head + EDL loss, Eq 5–9).
- **Reported tables**: Tables III (segmentation), IV (detection + grading), V (calibration), VI (transfer ablation), VII (GNN ablation), VIII (component ablation), IX (zero-shot SKI10), X (compute cost). Table II catalogues datasets.
- **Reported figures**: Fig 1 (architecture), Fig 2 (mathematical principle diagram), Fig 3 (qualitative seg + uncertainty maps), Fig 4 (chord + box + Sankey + heatmap analyses).
- **Reported ablations**: 6 transfer-mechanism variants (Table VI), 5 GNN-architecture variants (Table VII), 7 component-removal variants (Table VIII).
- **Code-availability statement** (verbatim, Sec "Code Availability"):
  > Code for CartiDT, including training scripts, evaluation pipelines, and pre-trained model checkpoints, will be released upon acceptance. The repository includes full configuration files sufficient to reproduce the main results reported in Tables III–X.
- **Released checkpoints**: planned but not yet released; the Zenodo / HuggingFace URL will be added to the README at acceptance.
- **Ethics**: existing de-identified data only; DUA 2024-0347; no new IRB review required (Ethics Statement ¶2).
- **Funding**: NSFC No. 81800834.

## 10. Notes for downstream turns

- The "no inline comment" rule is honoured project-wide. Module-level docstrings carry the paper anchors instead.
- The document header is intentionally a plain `# Project Context` — no file-path provenance string is leaked into any committed document.
- Module B's edge weights are **frozen** after a one-time pre-pass on the mean training segmentation. This is the only training-data leakage allowance the manuscript makes (Sec III.D-2). Document it in `docs/deviations.md` if any implementation detail varies.
