# Experiment and reproduction record

## Selected run

The fixed best audio-visual run is `hl-video_tef-exp-2025_10_23_23_51_04`, checkpoint epoch 199 (zero-based), seed 2017. The matching code was recovered from that run's saved archive, rather than taking a later working-directory variant. The original checkpoint SHA-256 is `d7e89805d8e90b071ddb10937a387abf12f52e73567a62b7e6417163a566b74a`.

The release exports the exact model tensors to a state-only checkpoint. Its new file hash is recorded in `assets-manifest.json`; tensor equality is checked during export. Repackaging removes optimizer and Python-object metadata without changing model values.

## Complete validation result

| Metric | Value |
| :--- | ---: |
| MR R1 @ 0.5 | 64.13 |
| MR R1 @ 0.7 | 48.06 |
| MR full average mAP | 43.63 |
| MR mAP @ 0.5 | 64.57 |
| MR mAP @ 0.75 | 43.97 |
| MR long / middle / short mAP | 51.55 / 44.68 / 9.21 |
| HL Fair mAP / Hit@1 | 75.52 / 75.68 |
| HL Good mAP / Hit@1 | 64.83 / 74.00 |
| HL Very Good mAP / Hit@1 | 39.72 / 63.61 |

Inference was rerun for **1,550 queries across 1,519 videos**. All 14 summary metrics matched the saved evaluation at its reported precision. `scripts/verify_metrics.py` independently recomputes them from published predictions and validation annotations without needing the model weights or feature files.

Average MR mAP follows the supplied QVHighlights evaluator over temporal IoU thresholds 0.50–0.95. Recall and highlight metrics follow the same evaluator. All values above are percentages on the validation split.

## Baseline comparison

The recorded local QD-DETR audio baseline has average MR mAP **41.72**, versus **43.63** for OCS-DETR: **+1.91 percentage points**. The baseline uses seed 3 and differs in positional/configuration choices from the final run, so this is a full-system comparison. It does not isolate the causal contribution of OT or CMAF. The best validation checkpoint is selected from experiments; no test-set or multi-seed superiority claim is made.

The highlight trade-off is visible in the record: the baseline's Very Good Hit@1 is 65.03, while the selected final run's is 63.61. The project improves its selected retrieval objective rather than every metric simultaneously.

## Public reproduction levels

1. **Metric recomputation:** all 1,550 saved predictions and labels, no GPU required.
2. **Model smoke run:** released model tensors and eight-query feature sample, suitable for CPU.
3. **Full model inference:** official full feature sets, the same checkpoint, and all validation annotations.

An eight-query smoke result verifies loading and execution; it is not the full validation benchmark.

For portability, decoder zeros are allocated on the input device rather than a hard-coded CUDA device. This preserves the GPU computation and permits CPU smoke inference. The training-only torchvision import is lazy; model parameters and forward mathematics are unchanged.

## Public runtime verification

The tensor-only release was also evaluated on CPU with PyTorch 2.14.1: all 1,550 queries completed, with strict checkpoint loading and all 14 summary metrics reproduced. See [the execution record](../evaluation/public_runtime_run.json). The recorded 16.144 seconds measures inference on the validation machine with precomputed features; it excludes feature extraction and is not a portable latency benchmark.
