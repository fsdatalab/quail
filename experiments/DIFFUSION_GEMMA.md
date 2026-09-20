# DiffusionGemma experiments

These scripts use the Quail implementation at commit `2d9cad1136cd21a29efe638d6ca9f9267790c950`.
They are archived here so the model PR can keep its runtime and correctness tests separate from experiment code.

To reproduce an experiment, create a worktree at that commit. It includes these scripts, the image helpers, and the reference configuration with a 256-token canvas.

```sh
git worktree add --detach /tmp/quail-diffusion-experiment 2d9cad1136cd21a29efe638d6ca9f9267790c950
cd /tmp/quail-diffusion-experiment
```

Follow the command in each script's module docstring. Run inference on Modal and tee its logs. Results belong on the `quail-results` volume.

- `diffusion_gemma_confirmation.py` checks filter and join execution and compares canvas configurations.
- `diffusion_gemma_layer_timing.py` measures individual model operations and expert kernel settings.
- `diffusion_gemma_readout_probe.py` compares answer scores under different submission and canvas settings.
- `quailb_sol.py` computes the speed-of-light estimates from corpus token counts without inference.
