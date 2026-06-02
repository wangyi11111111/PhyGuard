# Open Source Notes

This repository is being prepared for paper submission and public release.

## What is original here

- PhyGuard framing: physics as a local reliability guard rather than a fixed
  global physics loss.
- Physics residual bank and local guarded correction protocol.
- Failure-mode-aware evaluation protocol for sparse, noisy, and disrupted
  traffic state reconstruction.
- Experiment aggregation, ablation, complexity, and interpretability tooling.

## Third-party components

The current research prototype evaluates PhyGuard on top of a strong
spatiotemporal reconstruction core and compares against common baselines
including BRITS, SAITS, GRINLite, MagiNet, and ImputeFormer.

Before final public release, verify and document the licenses for:

- any official or reconstructed MagiNet/GRIN code used in experiments;
- PyPOTS implementations of BRITS, SAITS, and ImputeFormer;
- public traffic datasets such as PEMS and METR-LA.

## Paper result caution

The README reports the current target-region masked MAE tables generated for
paper planning. These should be re-run on the final release branch before
camera-ready submission.

