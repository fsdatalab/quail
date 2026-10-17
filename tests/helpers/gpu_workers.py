"""CPU processes implementing the GPU worker message protocol."""


def run(gpu_index, connection):
    model = None
    boot_count = 0
    while True:
        try:
            kind, data = connection.recv()
        except EOFError:
            return
        if kind == "boot":
            boot_count += 1
            cold = model is None
            if cold:
                model = data["model"]
            result = {"boot_s": 0.0, "kind": "cold" if cold else "warm",
                      "boot_count": boot_count}
        elif kind == "fail" and gpu_index == 0:
            connection.send(("err", "worker failed"))
            continue
        elif kind == "exit" and gpu_index == 0:
            return
        else:
            result = {"model": model, "value": data}
        connection.send(("ok", result))
