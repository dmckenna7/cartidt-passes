# Repo Plan

## Directory tree (final)

```
cartidt/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── main.yaml
│   ├── ablation_transfer_full_ft.yaml
│   ├── ablation_transfer_slice_wise_late.yaml
│   ├── ablation_transfer_patch3d.yaml
│   ├── ablation_transfer_lora_r4.yaml
│   ├── ablation_transfer_lora_no_depth.yaml
│   ├── ablation_transfer_lora_r64.yaml
│   ├── ablation_gnn_indep_mlp.yaml
│   ├── ablation_gnn_concat_mlp.yaml
│   ├── ablation_gnn_self_attn.yaml
│   ├── ablation_gnn_fc_graph.yaml
│   ├── ablation_gnn_biomech.yaml
│   ├── ablation_component_no_lora.yaml
│   ├── ablation_component_no_depth_interp.yaml
│   ├── ablation_component_no_gnn.yaml
│   ├── ablation_component_softmax_head.yaml
│   ├── ablation_component_fc_edges.yaml
│   ├── ablation_component_uniform_edges.yaml
│   ├── ablation_component_no_joint.yaml
│   ├── transfer_ski10.yaml
│   └── _unittest.yaml
├── cartidt/
│   ├── __init__.py
│   ├── link.py
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── catalog.py
│   │   ├── sources/{__init__.py, oai_dess.py, ski10.py, iwoai_split.py}
│   │   └── conditioning/{__init__.py, spatial_aug.py, intensity_norm.py}
│   ├── passes/
│   │   ├── __init__.py
│   │   ├── encode/{__init__.py, vit_adapter.py, dinov2_weights.py, lora.py, depth_posembed.py}
│   │   ├── segment/{__init__.py, upernet.py, pyramid.py, pooling.py}
│   │   ├── twin/{__init__.py, graph_sage.py, anatomy_graph.py, contact_weights.py, region_features.py}
│   │   └── evidence/{__init__.py, dirichlet_head.py, dirichlet.py, decompose.py, edl.py}
│   ├── diagnostics/{__init__.py, overlap.py, ordinal.py, calibration.py, significance.py, reference_models.py, cost.py, selective.py}
│   ├── backend/{__init__.py, trainer.py, objective.py, segmentation_loss.py, optimizer.py, schedule.py, checkpointing.py, reproducibility.py, distributed.py}
│   └── driver/{__init__.py, train.py, evaluate.py, infer.py, export_onnx.py, panel_uncertainty.py, panel_analysis.py}
├── scripts/{launch_train.sh, launch_eval.sh, prepare_oai.sh}
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── shape/{__init__.py, test_contracts.py, test_pos_depth_shape.py, test_upernet_shape.py, test_topology.py, test_node_features_shape.py, test_sage_shape.py, test_evidence_head_shape.py, test_oai_loader.py, test_ski10_loader.py, test_iwoai_loader.py, test_registry.py, test_augment_shape.py, test_baseline_registry.py, test_flop_count.py}
│   ├── regression/{__init__.py, test_lora_freeze_signature.py, test_edge_weights_fixed.py, test_dirichlet_pdf_vs_scipy.py, test_uncertainty_closed_form.py, test_edl_loss_vs_sensoy.py, test_soft_dice_vs_monai.py, test_dsc_vs_monai.py, test_auroc_vs_sklearn.py, test_ece_curve.py, test_bootstrap_known_input.py, test_scheduler_curve.py, test_determinism.py}
│   ├── overfit/{__init__.py, test_single_batch_overfit.py}
│   └── integration/{__init__.py, test_four_step_pipeline.py, test_config_inflation.py, test_ski10_transfer_glue.py}
├── docs/{project-context.md, implementation-map.md, repo-plan.md, deviations.md}
└── assets/{fig1_architecture.svg, fig2_principle.svg}
```

The package is organised as a compiler-style pass pipeline: a thin `link.py` composes the
network and exposes `build_model`; `frontend/` lowers raw volumes into the typed batch IR;
`passes/` holds the four model passes that successively rewrite that IR (`encode`, `segment`,
`twin`, `evidence`); `backend/` lowers the composed model to a trained checkpoint;
`diagnostics/` are the read-only analysis passes; and `driver/` is the command-line front end.

## Module-level responsibilities

- **frontend**: typed batch contracts (`schema`); the dataset factory (`catalog`); loaders for OAI, OAI-ZIB, IWOAI, SKI10 under `sources/`; per-volume z-score normalisation and the six augmentation operators from §V.C under `conditioning/`.
- **passes/encode**: load `facebook/dinov2-large` (`dinov2_weights`); insert LoRA rank-16 on Q and V (`lora`); build the depth-interpolated positional embedding (`depth_posembed`); the wrapped backbone (`vit_adapter`) exposes `forward(volume) -> list[FeatureMap]` at four pyramid scales for UPerNet.
- **passes/segment**: UPerNet — pyramid pooling on the deepest feature (`pooling`), FPN top-down (`pyramid`), project to 7 logits at native MRI resolution (`upernet`).
- **passes/twin**: 14-node topology (`anatomy_graph`); one-shot edge-weight estimator over the mean training seg (`contact_weights`); per-subject masked global-average node features (`region_features`); 3-layer GraphSAGE aggregation with edge-scaled mean (`graph_sage`).
- **passes/evidence**: MLP → α (`dirichlet_head`); the Dirichlet PDF (`dirichlet`); closed-form aleatoric / epistemic (`decompose`); Eq (9) EDL loss with linear-annealed KL (`edl`).
- **diagnostics**: DSC / ASSD (`overlap`); AUROC / κ_w (`ordinal`); ECE / AURC / FPR@τ (`calibration`); selective-prediction sweep (`selective`); paired bootstrap CI + Holm–Bonferroni (`significance`); baseline registry (`reference_models`); param + FLOP + per-volume timing counter (`cost`).
- **backend**: AdamW (`optimizer`), cosine + linear-warmup schedule (`schedule`), joint loss (`objective`), soft-Dice + CE seg loss (`segmentation_loss`), grad-accum epoch loop (`trainer`), deterministic set-up (`reproducibility`), atomic checkpoint (`checkpointing`), DDP launcher (`distributed`).
- **driver**: CLI entry points `train`, `evaluate`, `infer`, `export_onnx`, plus two figure-rendering scripts (`panel_uncertainty`, `panel_analysis`) for Figs 3–4.

## Pinned dependencies (`requirements.txt`)

```
torch==2.1.2
torchvision==0.16.2
timm==0.9.16
einops==0.7.0
omegaconf==2.3.0
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.1
monai==1.3.0
nibabel==5.2.0
SimpleITK==2.3.1
tqdm==4.66.2
tensorboard==2.16.2
pandas==2.1.4
matplotlib==3.8.2
```

Development extras (pyproject `[project.optional-dependencies] dev`):

```
ruff==0.3.4
black==24.3.0
isort==5.13.2
mypy==1.9.0
pytest==8.1.1
pytest-xdist==3.5.0
pre-commit==3.6.2
```

## Expected test coverage

- `tests/shape/` — 14 files; covers every public function in every component module, asserting output tensor shape, dtype, and device.
- `tests/regression/` — 12 files; cross-checks our implementations against `scipy`, `sklearn`, `monai`, and against numeric snapshots.
- `tests/overfit/` — 1 file; single 32³ synthetic volume, 200 optimiser steps; asserts `L_seg < 0.05`.
- `tests/integration/` — 3 files; 4-step end-to-end with DDP disabled; cross-check that each ablation YAML inflates into a runnable model; SKI10 zero-shot evaluate-only glue.

CI on every push: `ruff check . && mypy --strict cartidt && pytest tests/shape tests/regression -q -n auto` (overfit + integration tests are gated behind `pytest --runslow`).
