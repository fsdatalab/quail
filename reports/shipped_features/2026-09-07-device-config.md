# Hardware selection in EngineConfig

GPU count, model, backend, and device now share one configuration:

```python
session = quail.Session(quail.EngineConfig(gpus=1, device="h100-sxm"))
```

`Session(device=...)` is replaced by `Session(EngineConfig(device=...))`.
Session arguments after `config` are keyword only. The session uses the same
hardware configuration on a GPU host or inside a Modal GPU function.
The resolved hardware specification remains available as `session.device`.

The default remains one H100. Selecting a device names the hardware used for
planning; it does not attach a GPU to a local process. CPU tests check device
selection, planning, and validation.
