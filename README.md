# Face-Classification-Verification-using-CNN
Homework 2 Part 2 for the course 11-785 Introduction to Deep Learning at CMU

# HW2P2: Face Classification and Verification

Course assignment (11-785, Spring 2026) on convolutional neural networks for face recognition. A custom CNN is trained to classify faces across 8,631 identities, then reused as a feature extractor to produce face embeddings. Pairs of embeddings are compared to decide whether two images show the same person, which solves the verification task.

## Two Tasks

- **Classification**: identify which of 8,631 known identities a face belongs to.
- **Verification**: given two images (possibly of identities never seen in training), decide same person or different person by comparing their embeddings. Scored by cosine similarity, with metrics including accuracy, EER, AUC, and TPR at fixed FPR.

## Data Pipeline

- **Source**: Kaggle competition `11785-hw-2-p-2-face-verification-spring-2026`. Dataset folder `hw2p2_data` contains `cls_data` (train/dev/test for classification) and `ver_data` plus `val_pairs.txt` / `test_pairs.txt` for verification.
- **Classification dataset** (`ImageDataset`): reads an `images/` folder and a `labels.txt` mapping; supports labeled and unlabeled partitions. Asserts that train, dev, and test share the same class set.
- **Verification dataset** (`ImagePairDataset`): reads a pairs file with `img1 img2 [label]` lines and returns image pairs (with the match label when present).
- **Transforms**: resize to 112x112, convert to float tensor scaled to [0, 1], normalize to mean/std 0.5 per channel. Training adds RandomHorizontalFlip (p=0.5), ColorJitter (brightness/contrast 0.2), RandomRotation (10 degrees), and RandomErasing (p=0.1). Validation and test use no augmentation.
- **DataLoaders**: train shuffled; validation and test not shuffled (test order must match Kaggle submission IDs).

## Model

A custom ResNet-50 built from `Bottleneck` blocks (1x1, 3x3, 1x1 convolutions with a projection shortcut), implemented from scratch in PyTorch. Block layout is the standard ResNet-50: stages of 3, 4, 6, 3 bottleneck blocks at increasing channel widths, followed by adaptive average pooling and a flatten.

- **Embedding head**: a linear layer projecting to a 512-dimensional embedding, with BatchNorm1d on the embedding.
- **Output**: the forward pass returns a dict with `feats` (512-D embedding) and, in the classification variant, `out` (class logits).
- **Constraint**: under the 30M trainable-parameter budget for this assignment.

Two model variants appear:
- `Network` includes a classification head over 8,631 classes (used for Phase 1).
- `NetworkPhase2` is the same backbone without a classification head, returning only the 512-D embedding (used for Phase 2 / verification).

## Training Strategy

Training is staged.

**Phase 1 (classification)**
- Loss: CrossEntropyLoss.
- Optimizer: SGD (lr 0.1, momentum 0.9, weight_decay 5e-4).
- Scheduler: CosineAnnealingLR.
- Mixed-precision training via `torch.amp.autocast` and `GradScaler`.
- Validated each epoch on classification top-1 accuracy and on verification metrics over the validation pairs.

**Phase 2 (ArcFace fine-tuning)** — present in the more complete notebook (`...2_6.ipynb`)
- Loads the best Phase 1 checkpoint into `NetworkPhase2`.
- Loss: ArcFaceLoss from `pytorch_metric_learning` (margin 28.6 degrees, scale 64.0) applied to the normalized embeddings.
- Optimizer: a fresh SGD (lr 0.01, momentum 0.9, weight_decay 5e-4) optimizing both the model and the ArcFace loss parameters.
- Scheduler: CosineAnnealingLR.
- Selects the best model by lowest verification EER on the validation pairs.

The angular-margin ArcFace loss sharpens the embedding space so that same-identity faces cluster tightly and different identities separate, which directly improves verification rather than only classification accuracy.

## Verification Scoring

For each pair, both images pass through the model, embeddings are L2-normalized, and cosine similarity gives the match score. `verification_metrics` converts scores and labels into accuracy, EER, AUC, and TPR at target FPRs. At test time, similarity scores are written out for Kaggle submission.

## Config Summary

| Setting | Value |
|---|---|
| Image size | 112 x 112 |
| Batch size | 128 |
| Num classes | 8,631 |
| Embedding size | 512 |
| LR (Phase 1) | 0.1 (SGD, momentum 0.9, wd 5e-4) |
| LR (Phase 2 ArcFace) | 0.01 |
| Epochs | 25 (more recommended beyond the early submission) |
| ArcFace margin / scale | 28.6 deg / 64.0 |
| Augmentation | flip, color jitter, rotation, random erasing |
| Precision | mixed (AMP) |

## Running the Notebook

- **Environments supported**: Colab, Kaggle, and PSC Bridges-2. Each has its own setup cells; set `config['data_root']` to match (`/content/dataset/hw2p2_data`, `/kaggle/input/.../hw2p2_data`, or `/local/dataset/hw2p2_data`).
- **Credentials**: requires a Kaggle API token (new `KGAT_` format) for data download and submission, and a WandB API key for logging and checkpoint restore.
- **PSC note**: download the dataset to the node-local `$LOCAL` storage to avoid shared-filesystem I/O bottlenecks; the local copy is cleared when you change nodes, so re-download per node.
- **Checkpoints / WandB**: `save_model` / `load_model` checkpoint the model, optimizer, scheduler, and metrics. The notebook can restore a prior best checkpoint from WandB (project `sddabir-carnegie-mellon-university/hw2p2`) instead of retraining; Phase 2 starts from the best Phase 1 weights.
- **Multi-GPU**: wraps the model in `DataParallel` when more than one GPU is available.

## Submission Deliverables

- `submission.csv` of verification similarity scores for the Kaggle private leaderboard.
- A completed `README` cell, packaged with the acknowledgement and WandB logs into the final submission zip for Autolab.

## Notes on the Two Notebook Versions

- `s26-hw2p2-starter-notebook2_6.ipynb` is the fuller version: ResNet-50 backbone with both Phase 1 (CrossEntropy classification) and Phase 2 (ArcFace embedding fine-tuning), including the ArcFace config and the WandB checkpoint-restore flow.
- `s26-hw2p2-starter-notebook__1_.ipynb` is an earlier, Phase-1-only version: the same ResNet-50 backbone trained with CrossEntropy, no ArcFace config or fine-tuning stage, and a partially filled README.

## Rules

No pretrained or pre-loaded models; all architecture is implemented from scratch with base PyTorch layers. No external data. Test order must not be shuffled so predictions align with Kaggle IDs.
