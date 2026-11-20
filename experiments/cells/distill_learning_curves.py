"""Learning curves of a Qwen3 0.6B student on a Qwen3.8 27B teacher's filter.

The teacher's labels, margins, and answer-row states come from an
answer trace of IMDB-1 at scale factor 0.5 (`QUAIL_ANSWER_TRACE`). A
fixed 5,000 reviews are the test set; each training set is a random
draw of 100, 300, 1,000, or 3,000 of the others. Every setup is scored
by agreement with the teacher's label on the test set, and by AUC.

Setups:
    zero_shot: the student's TRUE minus FALSE logit on the teacher's
        prompt, untrained.
    bits: LoRA on the teacher's labels.
    margin: LoRA on the teacher's probability, sigmoid(margin).
    margin_state: margin, plus a projection of the student's answer-row
        state onto the teacher's, read with the teacher's answer
        direction (SimKD).
    document_probe: LoRA and a linear probe on the student's state at
        the review's last token, with no question in the input.
    frozen_answer_probe, frozen_document_probe: a linear probe on the
        untrained student's answer-row or last-review-token state.

Prediction: zero_shot agrees about 80 percent. At 3,000 labels bits,
margin, and margin_state reach 93 to 95 percent; margin leads bits by
one to two points at 100 and 300 labels. The frozen answer probe sits
three to five points under LoRA. The document probes trail the
question-reading setups by one to three points.

    uv run modal run --detach -m experiments.cells.distill_learning_curves \
        2>&1 | tee "$run_log"

Records: /results/distillation/learning_curves/<run id>.json
"""

import json
import os
import time

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image(packages=("peft==0.22.0",)).add_local_python_source(
        "experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/results": results}

STUDENT = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TEACHER = "Qwen/Qwen3.8-27B-FP8"
TEACHER_REVISION = "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"
TRACE_RUN = "/results/distillation/answer_trace/20261119T221615Z-db4b2ebe"
TRACE_DIR = f"{TRACE_RUN}/answer_trace/imdb/IMDB-1"
PROMPT_PIECES = f"{TRACE_RUN}/quail/imdb/IMDB-1/prompt_pieces.json"
SCALE_FACTOR = 0.5
NUM_TEST = 5000
DRAWS = {100: (0, 1, 2), 300: (0, 1, 2), 1000: (0,), 3000: (0,)}
TRAINED = ("bits", "margin", "margin_state", "document_probe")
LORA_RANK = 16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
MICRO_DOCUMENTS = 4
ACCUMULATE = 4
EPOCHS = 4
MIN_STEPS, MAX_STEPS = 60, 800
EVAL_TOKENS = 65536


def _load():
    """Return the prompts, the teacher's margins and states, and the splits."""
    import glob

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from quail_b.data import load_table

    trace_path = glob.glob(f"{TRACE_DIR}/*.parquet")[0]
    trace = pq.read_table(trace_path).to_pandas()
    states = np.load(trace_path.replace(".parquet", ".npy"))
    bodies = load_table("reviews", scale_factor=SCALE_FACTOR).column(
        "body").to_pylist()
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER, revision=TEACHER_REVISION)
    with open(PROMPT_PIECES) as pieces_file:
        pieces = json.load(pieces_file)
    preamble = teacher_tokenizer.decode(pieces["preamble"])
    tail = teacher_tokenizer.decode(pieces["filters"][0]["tail"])
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, revision=STUDENT_REVISION)
    head = tokenizer.encode(preamble, add_special_tokens=False)
    tail_tokens = tokenizer.encode(tail, add_special_tokens=False)
    documents = [head + tokenizer.encode(bodies[index], add_special_tokens=False)
                 for index in trace.document]
    prompts = [document + tail_tokens for document in documents]
    margins = trace.margin.to_numpy(np.float32)
    order = np.random.default_rng(0).permutation(len(margins))
    test, pool = order[:NUM_TEST], order[NUM_TEST:]
    true_id, = tokenizer.encode("TRUE", add_special_tokens=False)
    false_id, = tokenizer.encode("FALSE", add_special_tokens=False)
    return dict(prompts=prompts, documents=documents, margins=margins,
                states=states, test=test, pool=pool,
                answer_ids=(true_id, false_id), pad_id=tokenizer.pad_token_id)


def _draw(pool, num_labels, seed):
    import numpy as np
    return np.random.default_rng(1000 + seed).choice(pool, num_labels,
                                                     replace=False)


def _auc(scores, positive):
    """Return the probability a positive scores above a negative."""
    import pandas as pd
    ranks = pd.Series(scores).rank().to_numpy()
    num_positive = int(positive.sum())
    num_negative = len(positive) - num_positive
    return float((ranks[positive].sum() - num_positive * (num_positive + 1) / 2)
                 / (num_positive * num_negative))


def _metrics(scores, data):
    import numpy as np
    margins = data["margins"][data["test"]]
    positive = margins > 0
    return {"agreement": float(((scores > 0) == positive).mean()),
            "auc": _auc(scores, positive),
            "pearson_margin": float(np.corrcoef(scores, margins)[0, 1])}


def _padded(torch, sequences, pad_id):
    """Right-pad token lists; return ids, mask, and each last position."""
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for position, sequence in enumerate(sequences):
        ids[position, :len(sequence)] = torch.tensor(sequence)
        mask[position, :len(sequence)] = 1
    last = mask.sum(1) - 1
    return ids.cuda(), mask.cuda(), last.cuda()


def _final_states(torch, backbone, sequences, pad_id):
    """Return the final-norm state at each sequence's last token."""
    ids, mask, last = _padded(torch, sequences, pad_id)
    hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    return hidden[torch.arange(len(sequences), device=hidden.device), last]


def _eval_groups(lengths, indices):
    """Split indices, sorted by length, into groups under EVAL_TOKENS padded."""
    ordered = sorted(indices, key=lambda index: lengths[index])
    groups, current = [], []
    for index in ordered:
        if current and lengths[index] * (len(current) + 1) > EVAL_TOKENS:
            groups.append(current)
            current = []
        current.append(index)
    return groups + [current] if current else groups


def _student(torch):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=STUDENT_REVISION, torch_dtype=torch.bfloat16).cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _answer_direction(model, answer_ids):
    weight = model.get_output_embeddings().weight
    true_id, false_id = answer_ids
    return (weight[true_id] - weight[false_id]).float()


def _teacher_direction(data):
    """Fit the teacher's TRUE minus FALSE head row from states and margins."""
    import numpy as np
    pool = data["pool"]
    direction, *_ = np.linalg.lstsq(
        data["states"][pool].astype(np.float32), data["margins"][pool],
        rcond=None)
    return direction


def _score_all(torch, data, inputs, indices, score):
    """Score inputs[index] for each index, in length-sorted groups."""
    import numpy as np
    lengths = [len(sequence) for sequence in inputs]
    out = np.zeros(len(indices), dtype=np.float32)
    where = {index: position for position, index in enumerate(indices)}
    with torch.no_grad():
        for group in _eval_groups(lengths, indices):
            values = score([inputs[index] for index in group])
            out[[where[index] for index in group]] = values.float().cpu().numpy()
    return out


@app.function(image=image, gpu="H100!", memory=98304, timeout=3 * 3600,
              volumes=VOLUMES)
def frozen() -> list:
    """Score the zero-shot student and fit probes on its frozen states."""
    import torch

    data = _load()
    model = _student(torch).eval()
    backbone = model.model
    direction = _answer_direction(model, data["answer_ids"])
    started = time.perf_counter()
    features = {}
    for name, inputs in (("answer", data["prompts"]),
                         ("document", data["documents"])):
        features[name] = _features(torch, backbone, inputs, data["pad_id"])
    feature_seconds = time.perf_counter() - started
    records = []
    zero_shot = (features["answer"][data["test"]] @ direction.cpu().numpy())
    records.append(dict(setup="zero_shot", num_labels=0, seed=0,
                        **_metrics(zero_shot, data)))
    for name in ("answer", "document"):
        for num_labels, seeds in DRAWS.items():
            for seed in seeds:
                train = _draw(data["pool"], num_labels, seed)
                started = time.perf_counter()
                scores = _probe(torch, features[name], data["margins"], train,
                                data["test"])
                records.append(dict(
                    setup=f"frozen_{name}_probe", num_labels=num_labels,
                    seed=seed, train_seconds=time.perf_counter() - started,
                    **_metrics(scores, data)))
    for record in records:
        record["feature_seconds"] = feature_seconds
    return records


def _features(torch, backbone, inputs, pad_id):
    """Return the final-norm last-token state of every input, as float32."""
    import numpy as np
    lengths = [len(sequence) for sequence in inputs]
    out = np.zeros((len(inputs), backbone.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for group in _eval_groups(lengths, range(len(inputs))):
            states = _final_states(torch, backbone,
                                   [inputs[index] for index in group], pad_id)
            out[group] = states.float().cpu().numpy()
    return out


def _probe(torch, features, margins, train, test):
    """Fit a standardized linear probe to sigmoid(margin); score the test set."""
    x = torch.tensor(features[train], device="cuda")
    mean, scale = x.mean(0), x.std(0) + 1e-4
    x = (x - mean) / scale
    target = torch.sigmoid(torch.tensor(margins[train], device="cuda"))
    weight = torch.zeros(x.shape[1], device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=500,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x @ weight + bias, target) + 1e-2 * weight.square().sum() / len(train)
        loss.backward()
        return loss

    optimizer.step(closure)
    x_test = (torch.tensor(features[test], device="cuda") - mean) / scale
    with torch.no_grad():
        return (x_test @ weight + bias).cpu().numpy()


@app.function(image=image, gpu="H100!", memory=98304, timeout=3 * 3600,
              volumes=VOLUMES)
def trained(setup: str) -> list:
    """Train one LoRA setup at every label count and draw; score each."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    data = _load()
    teacher_direction = torch.tensor(_teacher_direction(data), device="cuda")
    states = data["states"]
    inputs = data["documents"] if setup == "document_probe" else data["prompts"]
    records = []
    for num_labels, seeds in DRAWS.items():
        for seed in seeds:
            torch.manual_seed(seed)
            base = _student(torch)
            direction = _answer_direction(base, data["answer_ids"])
            model = get_peft_model(base, LoraConfig(
                r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0,
                target_modules=list(LORA_TARGETS)))
            backbone = model.get_base_model().model
            hidden = backbone.config.hidden_size
            projection = torch.nn.Linear(hidden, states.shape[1], bias=False,
                                         device="cuda")
            probe = torch.nn.Linear(hidden, 1, device="cuda")
            heads = {"margin_state": projection, "document_probe": probe}.get(setup)

            def score(final, heads=heads):
                if setup == "margin_state":
                    return heads(final.float()) @ teacher_direction
                if setup == "document_probe":
                    return heads(final.float())[:, 0]
                return final.float() @ direction

            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": 2e-4}]
            if heads is not None:
                groups.append({"params": list(heads.parameters()), "lr": 1e-3})
            optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            train = _draw(data["pool"], num_labels, seed)
            per_step = MICRO_DOCUMENTS * ACCUMULATE
            steps = int(np.clip(EPOCHS * num_labels // per_step, MIN_STEPS,
                                MAX_STEPS))
            rng = np.random.default_rng(seed)
            order = []
            model.train()
            started = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad()
                for _ in range(ACCUMULATE):
                    if len(order) < MICRO_DOCUMENTS:
                        order.extend(rng.permutation(train).tolist())
                    picked = [order.pop() for _ in range(MICRO_DOCUMENTS)]
                    final = _final_states(torch, backbone,
                                          [inputs[i] for i in picked],
                                          data["pad_id"])
                    logits = score(final)
                    margin = torch.tensor(data["margins"][picked], device="cuda")
                    target = ((margin > 0).float() if setup == "bits"
                              else torch.sigmoid(margin))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target)
                    if setup == "margin_state":
                        teacher = torch.tensor(states[picked], device="cuda").float()
                        loss = loss + ((projection(final.float()) - teacher)
                                       .square().sum(1).mean()
                                       / teacher.square().sum(1).mean())
                    (loss / ACCUMULATE).backward()
                optimizer.step()
            train_seconds = time.perf_counter() - started
            model.eval()
            started = time.perf_counter()
            scores = _score_all(torch, data, inputs, data["test"], lambda group:
                                score(_final_states(torch, backbone, group,
                                                    data["pad_id"])))
            records.append(dict(
                setup=setup, num_labels=num_labels, seed=seed, steps=steps,
                train_seconds=train_seconds,
                eval_seconds=time.perf_counter() - started,
                **_metrics(scores, data)))
            print(json.dumps(records[-1]), flush=True)
            del model, base, optimizer, projection, probe
            torch.cuda.empty_cache()
    return records


@app.function(image=image, timeout=4 * 3600, volumes=VOLUMES)
def run_all() -> str:
    """Run every setup, each in its own container, and save the records."""
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pending = frozen.spawn()
    records = [record for part in trained.map(TRAINED) for record in part]
    records = pending.get() + records
    path = f"/results/distillation/learning_curves/{run_id}-imdb1-sf05.json"
    results.reload()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as out:
        json.dump({"trace": TRACE_DIR, "student": STUDENT,
                   "student_revision": STUDENT_REVISION, "num_test": NUM_TEST,
                   "draws": {str(k): list(v) for k, v in DRAWS.items()},
                   "lora_rank": LORA_RANK, "epochs": EPOCHS,
                   "steps": [MIN_STEPS, MAX_STEPS], "records": records},
                  out, indent=1)
    results.commit()
    return path


@app.local_entrypoint()
def main():
    """Start the run and print its function call id."""
    call = run_all.spawn()
    print(f"function call id: {call.object_id} (learning curves)", flush=True)
