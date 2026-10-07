# Data and feature preparation

This project uses [QVHighlights](https://github.com/jayleicn/moment_detr), introduced by Lei et al. for natural-language moment retrieval and highlight detection. Validation annotations and the derived predictions are included to make the reported metrics inspectable.

For full inference, obtain the official feature archives using the maintained data links in [CG-DETR](https://github.com/wjun0830/CGDETR#setup). The older QD-DETR feature link is marked expired upstream. Audio PANN features follow the QD-DETR/UMT preparation. Observe the dataset and upstream distribution terms.

Place the validation features under:

```text
features/
  clip_features/        # video-id.npz
  slowfast_features/    # video-id.npz
  clip_text_features/   # qid.npz
  umt_pann_features/    # video-id.npy
```

The full validation copy contains 1,519 files in each video/audio directory and 1,550 query feature files. A small eight-query subset is provided as a release asset for the CPU smoke command, together with its sample annotation file. It contains precomputed numerical features and no raw video or personal data. Full feature archives are not committed to Git.

The interactive timeline viewer plots published query annotations and model predictions; it does not redistribute the original YouTube videos.
