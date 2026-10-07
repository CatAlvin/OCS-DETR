# Attribution

- **QD-DETR — baseline:** [wjun0830/QD-DETR](https://github.com/wjun0830/QD-DETR), Moon et al., CVPR 2023. The `qd_detr`, `utils`, and evaluator code retain the QD-DETR/Moment-DETR foundation and upstream notices. The original combined license is preserved in `LICENSE`.
- **TR-DETR — HD–MR task interaction:** [mingyao1120/TR-DETR](https://github.com/mingyao1120/TR-DETR), Sun et al., AAAI 2024. Its task-reciprocal interaction idea informed this project's further refinement. The upstream license is retained in `licenses/TR-DETR.txt`.
- **Moment-DETR / QVHighlights:** [jayleicn/moment_detr](https://github.com/jayleicn/moment_detr), Lei et al., NeurIPS 2021. Dataset annotations and evaluation foundation.
- **DETR:** Facebook Research; copyright headers are preserved in the inherited transformer/detection implementation.

Bohan Wu's project contributions are the gated transport alignment, query-conditioned fusion integration, proposal-weighted differentiable soft-window refinement, audio-visual experiments and checkpoint reproduction. Existing backbones, pretrained feature extractors, dataset annotations and evaluation methods are not claimed as original work.

## References

Moon et al. *Query-Dependent Video Representation for Moment Retrieval and Highlight Detection.* CVPR 2023. [Paper](https://arxiv.org/abs/2303.13874).

Sun et al. *TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection.* AAAI 2024. [Paper](https://arxiv.org/abs/2401.02309).

Lei et al. *Detecting Moments and Highlights in Videos via Natural Language Queries.* NeurIPS 2021. [Project](https://github.com/jayleicn/moment_detr).

The context-query attention building blocks in `qd_detr/interaction/test_CQA.py` follow [VSLNet](https://github.com/26hzhang/VSLNet). Their reuse and integration are distinguished from the project-specific OT and soft-window feedback modules; see [licenses/VSLNet.txt](licenses/VSLNet.txt).
