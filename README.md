# CartiDT — read the blueprint

CartiDT lowers a single 3D DESS knee volume through a fixed chain of **passes**, the way a
compiler lowers source to machine code. Each pass rewrites the data into a richer
intermediate form: raw voxels → encoded features → dense segmentation **and** a
biomechanical digital-twin graph → a Dirichlet belief over damage grade with a closed-form
split into aleatoric and epistemic uncertainty. The map below is the whole program; every
numbered block has its own section with the module that implements it, the command that
exercises it, and the number it is expected to produce.

```
            3D DESS knee volume  (.nii.gz, 160 x 384 x 384)
                       |
                       v
        +-------------------------------+
        | [1] frontend                  |   parse + z-score -> Batch IR
        +-------------------------------+
                       |  x : B x 1 x D x H x W
                       v
        +-------------------------------+
        | [2] passes/encode             |   frozen DINOv2 ViT-L
        |     LoRA(q,v) + depth posembed|   + depth-interpolated pos-embed
        +-------------------------------+
                       |  multi-scale feature pyramid
            +----------+-----------+
            v                      v
   +-----------------+    +--------------------+
   | [3] passes/     |    | [4] passes/twin    |   14-node, 26-edge
   |     segment     |    |   region features  |   GraphSAGE on the
   |   UPerNet       |    |   -> GraphSAGE      |   anatomical contact graph
   |   (FPN + PPM)   |    +--------------------+
   +-----------------+              |
            |  seg logits           |  per-compartment context
            |   (7 classes)         v
            |             +--------------------+
            +------------>| [5] passes/evidence|   Dirichlet head -> alpha
                          |   EDL + decompose  |   -> grade, u_ale, u_epi
                          +--------------------+
                                    |
                       +------------+------------+
                       v                         v
               +---------------+        +------------------+
               | backend       |        | [6] diagnostics  |  DSC ASSD AUROC
               |  L_seg + mu.L_EDL       |   + significance |  k_w ECE AURC FPR
               |  -> trained ckpt|       +------------------+
               +---------------+
                       |
                       v
        link.py composes [1..5];  driver runs: train | evaluate | infer | export_onnx
```

The tree mirrors the diagram one-for-one:

```
cartidt/
  link.py            build_model(cfg) -> CartiDT  (wires passes [1]-[5])
  frontend/          [1] schema, catalog, sources/{oai_dess,ski10,iwoai_split}, conditioning/
  passes/
    encode/          [2] vit_adapter, lora, depth_posembed, dinov2_weights
    segment/         [3] upernet, pyramid, pooling
    twin/            [4] anatomy_graph, contact_weights, region_features, graph_sage
    evidence/        [5] dirichlet, dirichlet_head, edl, decompose
  backend/           trainer, objective (L_seg + mu.L_EDL), optimizer, schedule, DDP
  diagnostics/       [6] overlap, ordinal, calibration, selective, significance, cost
  driver/            train, evaluate, infer, export_onnx + figure panels
```

<details>
<summary><b>Build it</b> — pip / conda / docker</summary>

```bash
# pip (editable, with dev tooling)
git clone https://github.com/dmckenna7/cartidt-passes.git && cd cartidt-passes
python -m pip install -e ".[dev]"

# conda
conda env create -f environment.yml && conda activate cartidt
python -m pip install -e .

# docker
docker build -t cartidt:latest .
docker run --gpus all -v /path/to/data:/workspace/cartidt/data cartidt:latest
```

`make help` lists the standard targets (`install dev lint type test smoke docker clean`).
The pinned toolchain is ruff 0.3.4 / black 24.3.0 / isort 5.13.2 / mypy 1.9.0 / pytest 8.1.1
on Python 3.10–3.11.
</details>

---

## [1] frontend — voxels become an IR

Reads a 3D DESS volume and its WORMS grading label, z-scores per volume, and packs everything
into the typed `Batch` that every later pass consumes. Dataset wiring (OAI-ZIB, IWOAI, SKI10)
lives in `frontend/catalog.py` and `frontend/sources/`.

- **Module:** `cartidt.frontend.schema`, `cartidt.frontend.catalog`, `cartidt.frontend.sources.*`
- **Smoke:** `python -m cartidt.driver.train --config configs/_unittest.yaml --out ./runs/smoke --steps 1`

## [2] passes/encode — frozen foundation model + LoRA

A frozen DINOv2 ViT-L is adapted with rank-16 LoRA on the query/value projections
(`alpha = 32`) and a depth-axis interpolated positional embedding so a 2D backbone reads 3D
slabs. Everything except the LoRA factors and the depth embedding stays frozen.

- **Module:** `cartidt.passes.encode.{vit_adapter,lora,depth_posembed,dinov2_weights}`
- **Paper anchor:** Eq. (1)–(2)

## [3] passes/segment — dense decoder

A UPerNet head (FPN + Pyramid Pooling) turns the feature pyramid into full-resolution
seven-class cartilage/meniscus logits.

- **Module:** `cartidt.passes.segment.{upernet,pyramid,pooling}`

## [4] passes/twin — biomechanical digital twin

Per-compartment features are pooled by masked global-average-pooling over the segmentation,
placed on a 14-node / 26-edge anatomical contact graph whose edge weights are estimated once
(offline) from the mean training segmentation, then refined by a hand-rolled GraphSAGE
(`L = 3`, `d_g = 256`).

- **Module:** `cartidt.passes.twin.{anatomy_graph,contact_weights,region_features,graph_sage}`
- **Paper anchor:** Eq. (3)–(4)

## [5] passes/evidence — Dirichlet belief + uncertainty

A two-layer evidential head emits Dirichlet concentrations (`softplus + 1`, `K = 4`). The EDL
loss anneals a KL-to-uniform term; `decompose` returns the closed-form aleatoric
`u_ale = (K / alpha0(alpha0+1)) . sum pi(1-pi)` and epistemic `u_epi = K / alpha0`, which drive
selective abstention.

- **Module:** `cartidt.passes.evidence.{dirichlet,dirichlet_head,edl,decompose}`
- **Paper anchor:** Eq. (5)–(9)

## backend — lowering to a trained checkpoint

The joint objective `L = L_seg + mu . L_EDL` (`mu = 0.5`, `L_seg` = soft-Dice + CE) is
optimised with AdamW under a cosine schedule with warmup; seeds {42, 123, 7}; DDP across 4
GPUs at effective batch 32.

- **Module:** `cartidt.backend.{objective,segmentation_loss,trainer,optimizer,schedule,reproducibility,distributed}`
- **Run:** `python -m cartidt.driver.train --config configs/main.yaml --out ./runs/main`
- **Ablations:** `python -m cartidt.driver.train --config configs/ablation_transfer_lora_no_depth.yaml --out ./runs/abl`

## [6] diagnostics — what each number should be

`evaluate` scores a checkpoint and writes the metric CSVs behind every paper table. Tolerances
over 3 seeds {42, 123, 7} on the IWOAI test split:

| Metric | Command | Expected |
| ------ | ------- | -------- |
| DSC mean (six compartments) | `evaluate --config configs/main.yaml` | 0.879 ± 0.002 |
| ASSD mean (mm) | `evaluate --config configs/main.yaml` | 0.33 ± 0.01 |
| Damage-detection AUROC | `evaluate --config configs/main.yaml` | 0.861 ± 0.003 |
| Quadratic-weighted κ_w | `evaluate --config configs/main.yaml` | 0.66 ± 0.01 |
| ECE | `evaluate --config configs/main.yaml` | 0.034 ± 0.002 |
| AURC | `evaluate --config configs/main.yaml` | 0.096 ± 0.004 |
| FPR @ 15% abstention | `evaluate --config configs/main.yaml` | 0.114 ± 0.005 |
| Zero-shot SKI10 DSC | `evaluate --config configs/transfer_ski10.yaml` | 0.842 ± 0.004 |

```bash
scripts/launch_eval.sh configs/main.yaml ./runs/main/last.ckpt ./eval/main
python -m cartidt.driver.evaluate --config configs/transfer_ski10.yaml --ckpt ./runs/main/last.ckpt --out ./eval/ski10
```

- **Module:** `cartidt.diagnostics.{overlap,ordinal,calibration,selective,significance,cost}`
- **Significance:** paired bootstrap `n = 10 000` + Holm–Bonferroni (`diagnostics.significance`)

---

## Where the data comes from

Four public cohorts, each behind its own DUA. Preprocess into NIfTI shards with
`scripts/prepare_oai.sh`, then place artefacts under `./data/oai_zib`, `./data/ski10`, and
`./data/manifests/{iwoai,ski10}/`. Expected on-disk size ≈ 80 GB plus checkpoints; a SHA-256 of
the canonical manifest tarball ships with each tagged release.

| Dataset | URL | License / DUA | Prep |
| ------- | --- | ------------- | ---- |
| OAI 3D DESS — 4 796 subjects → 3 212 / 802 / 782 grading split | <https://nda.nih.gov/oai> | NIH DUA — registration required | `scripts/prepare_oai.sh RAW_DIR OUT_DIR` |
| OAI-ZIB segmentations (507 vols) | distributed with OAI via ZIB | OAI DUA + ZIB redistribution terms | bundled with the OAI prep step |
| IWOAI 2019 partition (60-14-14 ×2 → 176 vols) | <https://github.com/IWOAI/knee-segmentation> | inherits OAI / OAI-ZIB | emitted under `manifests/iwoai/` |
| SKI10 (100 train / 50 test; femoral + tibial) | <http://www.ski10.org> | challenge-distribution terms | none — load with `SKI10Dataset` |

## What a full run costs

| Item | Value |
| ---- | ----- |
| Hardware | 4 × NVIDIA A100 80 GB, NVLink, CUDA 12.x |
| Main training wall-clock | 38 h to 120 epochs |
| All 18 ablations from scratch | ≈ 38 h × 18 |
| Per-volume inference (single A100) | 3.2 s full pipeline (2.7 s segmentation-only) |
| Storage | ≈ 80 GB processed dataset; ~1.2 GB per checkpoint |
| Effective batch | per-GPU 4 × grad-accum 2 × 4 GPUs = 32 |

## Ethics

Existing de-identified data only; NIH Data Use Agreement protocol **2024-0347**; no new IRB
review required. Funding: NSFC No. 81800834.

## Code availability

> Code for CartiDT, including training scripts, evaluation pipelines, and pre-trained model
> checkpoints, will be released upon acceptance. The repository includes full configuration
> files sufficient to reproduce the main results reported in Tables III–X.

Pre-trained checkpoints (3 seeds) will be released after acceptance at a stable URL with a
SHA-256 verification manifest; until then, weights are available upon reasonable request via
the corresponding author.

## Citation

```bibtex
@article{cartidt,
  title   = {Foundation Models Meet Musculoskeletal Digital Twins: Uncertainty-Aware Early Detection of Subclinical Cartilage Damage via Cross-Modal Transfer Learning},
  author  = {Feng, Haoxuan and Peng, Dangbing},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
}
```
