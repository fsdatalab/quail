"""Cluster the flagged steps (step 3) and write each run's agent inputs (steps 4 and 5).

Three clustering methods, each run twice (runs `<method>_a` and `<method>_b`):
- sae: NMF over the SAE features of each step's reasoning;
- text: NMF over TF-IDF word weights of each step's reasoning;
- random: each step goes into a random cluster.

For each cluster the agents see up to 10 sampled steps, spread evenly
over the cluster's strength (random: chosen at random), and the 2
strongest steps of the nearest cluster that are not in it.

    uv run --no-sync python -m experiments.safety.reward_hack_types.cluster WORK

Reads WORK/steps.json (from build_steps) and WORK/features.parquet (from
`modal_cells --step features`). Writes, per run, WORK/<run>/:
- open/<clusters|batches>.md and open/tasks.md, the open-coding agent's input;
- open/PROMPT.md, the open-coding agent's instructions (step 4);
- book/steps.json and book/README.md, the codebook agent's input;
- book/PROMPT.md, the codebook agent's instructions (step 5);
- meta.json, the steps the agents read and each step's cluster.

Run step 4 by giving a Claude agent open/PROMPT.md; copy its
open/open_codes.json to book/; then run step 5 with book/PROMPT.md.
"""

import argparse
import json
import os
import random
from string import Template

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_CLUSTERS = 10
PER_CLUSTER = 10
NEAR = 2
NUM_FEATURES = 65536
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def sae_matrix(steps: list[dict], path: str):
    """Return the idf-weighted SAE features of `steps`, each row scaled to length 1."""
    table = pq.read_table(path, columns=["s.id", "last_features"])
    position = {s: i for i, s in enumerate(table.column(0).to_pylist())}
    table = table.take([position[s["step_id"]] for s in steps])
    column = table["last_features"].combine_chunks()
    flat = column.flatten()
    matrix = sparse.csr_matrix(
        (
            flat.field("strength").to_numpy().copy(),
            flat.field("feature").to_numpy().copy(),
            column.offsets.to_numpy().copy(),
        ),
        shape=(len(steps), NUM_FEATURES),
    )
    idf = np.log(
        len(steps) / np.maximum(np.asarray((matrix > 0).sum(axis=0)).ravel(), 1)
    )
    weighted = matrix.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    return sparse.diags(1 / np.maximum(norms, 1e-9)) @ weighted


def text_matrix(steps: list[dict]):
    """Return TF-IDF word weights of each step's reasoning."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_]+\b",
    )
    return vectorizer.fit_transform([s["reasoning"] for s in steps])


def strengths(matrix) -> np.ndarray:
    """Return each step's share of strength on each NMF cluster."""
    weights = NMF(
        n_components=NUM_CLUSTERS, init="nndsvd", random_state=0, max_iter=1000
    ).fit_transform(matrix)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def sample(weights: np.ndarray, is_random: bool, seed: int):
    """Pick each cluster's sampled steps and contrast steps.

    Returns:
        (members, picked, contrast): step positions per cluster.
    """
    rng = random.Random(seed)
    labels = weights.argmax(axis=1)
    members = {
        k: [i for i in range(len(labels)) if labels[i] == k]
        for k in range(NUM_CLUSTERS)
    }
    picked, contrast = {}, {}
    for k, m in members.items():
        if is_random:
            picked[k] = rng.sample(m, min(PER_CLUSTER, len(m)))
        else:
            m = sorted(m, key=lambda i: -weights[i, k])
            picked[k] = (
                m
                if len(m) <= PER_CLUSTER
                else [
                    m[round(j * (len(m) - 1) / (PER_CLUSTER - 1))]
                    for j in range(PER_CLUSTER)
                ]
            )
    for k in range(NUM_CLUSTERS):
        if is_random:
            near = (k + 1) % NUM_CLUSTERS
            contrast[k] = rng.sample(members[near], min(NEAR, len(members[near])))
        else:
            near = max(
                (j for j in range(NUM_CLUSTERS) if j != k),
                key=lambda j: float(np.minimum(weights[:, k], weights[:, j]).sum()),
            )
            contrast[k] = sorted(members[near], key=lambda i: -weights[i, near])[:NEAR]
    return members, picked, contrast


def write_run(
    folder: str, steps: list[dict], weights: np.ndarray, is_random: bool, seed: int
) -> None:
    """Write one run's agent inputs and prompts into `folder`."""
    members, picked, contrast = sample(weights, is_random, seed)
    group, groups = ("batch", "batches") if is_random else ("cluster", "clusters")
    how = (
        f"The steps were split into {NUM_CLUSTERS} batches at random, "
        "so a batch may mix several kinds of steps."
        if is_random
        else f"A clustering method split the steps into {NUM_CLUSTERS} clusters. "
        "Each cluster shows up to "
        f"{PER_CLUSTER} sampled steps, spread from its strongest member "
        "to its weakest, "
        f"then {NEAR} steps "
        "from the nearest cluster that are not in it."
    )
    for sub in ("open", "book"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    blocks = []
    for k in range(NUM_CLUSTERS):
        order = (
            "chosen at random"
            if is_random
            else "spread evenly from the strongest to the weakest member"
        )
        other = "another batch" if is_random else "the nearest cluster"
        lines = [
            f"# {group.title()} {k + 1}\n{len(members[k])} steps in the {group}. "
            f"Below: {len(picked[k])} "
            f"sampled steps, {order}, then {len(contrast[k])} steps from {other} "
            f"that are not in this {group}.\n"
        ]
        for tag, chosen in (("Step", picked[k]), (f"Not in this {group}", contrast[k])):
            for j, i in enumerate(chosen, 1):
                s = steps[i]
                strength = (
                    ""
                    if is_random or tag != "Step"
                    else f", strength {weights[i, k]:.2f}"
                )
                lines.append(
                    f"### {tag} {j}: {s['step_id']} (agent {s['agent']}{strength})\n"
                    f"Task: {s['task']} (full text in tasks.md)\n"
                    f"Agent reasoning:\n{s['reasoning']}\n"
                    f"Commands:\n{s['commands'] or '(none)'}\n"
                )
        blocks.append("\n".join(lines))
    with open(os.path.join(folder, "open", f"{groups}.md"), "w") as out:
        out.write("\n".join(blocks))
    seen = sorted(
        {
            steps[i]["step_id"]
            for k in range(NUM_CLUSTERS)
            for i in picked[k] + contrast[k]
        }
    )
    by_id = {s["step_id"]: s for s in steps}
    tasks = {by_id[i]["task"]: by_id[i]["task_description"] for i in seen}
    with open(os.path.join(folder, "open", "tasks.md"), "w") as out:
        out.write("\n\n".join(f"## Task {t}\n{tasks[t]}" for t in sorted(tasks)))
    with open(os.path.join(folder, "book", "steps.json"), "w") as out:
        json.dump([by_id[i] for i in seen], out, ensure_ascii=False)
    with open(os.path.join(folder, "meta.json"), "w") as out:
        json.dump(
            {
                "seen": seen,
                "cluster": {
                    s["step_id"]: int(weights[i].argmax()) + 1
                    for i, s in enumerate(steps)
                },
            },
            out,
        )
    with open(os.path.join(PROMPTS, "context.md")) as source:
        context = source.read().strip()
    fill = {
        "folder": os.path.abspath(folder),
        "root": os.path.dirname(os.path.abspath(folder)),
        "group": group,
        "groups": groups,
        "how": how,
        "context": context,
        "num_groups": NUM_CLUSTERS,
    }
    for name, path in (
        ("open_coding.md", "open/PROMPT.md"),
        ("codebook.md", "book/PROMPT.md"),
        ("codebook_readme.md", "book/README.md"),
    ):
        with open(os.path.join(PROMPTS, name)) as source:
            text = Template(source.read()).substitute(fill)
        with open(os.path.join(folder, path), "w") as out:
            out.write(text)
    print(
        folder,
        "cluster sizes",
        sorted(
            np.bincount(weights.argmax(axis=1), minlength=NUM_CLUSTERS).tolist(),
            reverse=True,
        ),
        "agents read",
        len(seen),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="folder for inputs and outputs")
    work = parser.parse_args().work
    with open(os.path.join(work, "steps.json")) as source:
        steps = json.load(source)
    sae = strengths(sae_matrix(steps, os.path.join(work, "features.parquet")))
    text = strengths(text_matrix(steps))
    for name, weights, is_random, seed in (
        ("sae_a", sae, False, 0),
        ("sae_b", sae, False, 0),
        ("text_a", text, False, 0),
        ("text_b", text, False, 0),
    ):
        write_run(os.path.join(work, name), steps, weights, is_random, seed)
    for name, seed in (("random_a", 11), ("random_b", 12)):
        rng = random.Random(seed)
        weights = np.eye(NUM_CLUSTERS)[[rng.randrange(NUM_CLUSTERS) for _ in steps]]
        write_run(os.path.join(work, name), steps, weights, True, seed)


if __name__ == "__main__":
    main()
