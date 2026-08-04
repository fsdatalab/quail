# Next steps (ordered)

1. Re-baseline on the modern toolchain. Rebase the experiment image
   (CUDA 13 devel base, so kernels JIT) - the measured read ceiling
   moved from 80,556 to 97,220 tok/s with zero code changes. Re-run
   the headline cells: 10k chain (expect low 40s from 49.9), the
   32B ladder, one scaling point. Restate every claim against the
   new ceiling; keep the six-arm falsification table as evidence.
2. Shared-scan fidelity check. Rerun shared-vs-separate with the
   confident-flip discriminator (logprobs=2, 0.2-nat gap) before the
   5.09x multi-query claim ships. Near-ties only -> claim ships with
   measured tolerance; confident flips -> routing bug to find.
3. SGLang completeness arms: fp8-KV speed cell and a truth-scored
   accuracy run. Closes the port question with parity now known.
4. Planner refusal path: implement Algorithm 2's feasibility gate in
   plan_query (audit finding: unimplemented); then experiment E12.
5. Paper revision: fold in shared scans, the falsification and
   attribution story, fusion's documented negative (two-format
   evidence), the corrected ceiling language (achieved rate, never
   "floor"). Then the related-work reading pass with receipts
   (Sarathi, LOTUS/Palimpzest/DocETL, Liu et al., Hydragen, SGLang,
   Photon 2 as the megakernel end of the design spectrum).
6. Remaining roster: E6 fork/no-sharing baseline, E8 interactive
   mode boundary, E9 retention validation, E10 planner end-to-end,
   E11 wrong-prior re-planning, 32B planner assertions.
7. Parked with triggers: hybrid-attention models (Kimi K3-class)
   break uniform pages -> FlatKV-style allocator + per-layer rewind;
   fused speculation awaits an upstream cascade-kernel fix (our
   evidence chain is the bug report); joins design (task 14) after
   the paper's scan story is closed.
