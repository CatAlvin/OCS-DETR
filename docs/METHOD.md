# Method

The selected implementation uses QD-DETR as its baseline. QD-DETR provides the query-conditioned DETR backbone, moment prediction machinery and evaluation foundation. TR-DETR introduced reciprocal interaction between highlight detection (HD) and moment retrieval (MR); OCS-DETR builds on that idea and further develops the moment-to-highlight refinement path.

## Gated optimal-transport alignment

`OTAligner` projects video and text tokens, computes cosine-distance costs and applies 100 Sinkhorn iterations with regularization 0.1. Its transport plan creates aligned features for both modalities. A sigmoid gate mixes aligned and original video tokens; a learned residual scale adds aligned information to the text branch. The residual scale starts at zero and is not constrained to a positive value.

The implementation retains the selected run's masking and numerical behavior so the checkpoint can be reproduced exactly. It should not be interpreted as a newly modified, mask-renormalized transport solver.

## Query-conditioned fusion

`VSLFuser` in `qd_detr/interaction/test_CQA.py` combines the aligned video and query features before the QD-DETR encoder/decoder. This historical filename is retained to match the saved code snapshot; it is an implementation module rather than a test entry point.

## Moment-to-highlight feedback

The `CMAF` path takes multiple moment proposals, constructs differentiable soft windows and pools video features inside those windows. Proposal scores weight the pooled features and temporal prior. Attention then refines the video memory through a gated residual update before highlight prediction. This extends the TR-DETR interaction idea with proposal-weighted soft-window aggregation rather than claiming task reciprocity itself as a new invention.

## Inputs and fixed configuration

| Input | Dimensions |
| :--- | ---: |
| CLIP + SlowFast video features | 2,816 + 2 temporal endpoint features |
| CLIP query text | 512 |
| PANN audio features | 2,048 + 2 temporal endpoint features |
| Hidden representation | 256 |
| Trainable parameters | 8,969,258 |

The model consumes precomputed features. Full raw-video feature extraction is outside the evaluation command and is not included in model-only timing.
