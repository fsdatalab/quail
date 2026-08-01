"""Independent schedule checker (paper Appendix B).

Deliberately does NOT import docengine.costmodel: every formula here is
re-implemented from the paper text so solver and validator do not share the
objective implementation. Returns batch-indexed errors, not a boolean.

Replay model: the validator tracks, per document, the logical frontier z_i,
the resident prefix length r_i (task-first: progress under the current task
prefix; doc-first: document prefix), the pinned prompt set, and which filter
evaluations have completed. It recomputes U, A, K_L, K_W, K_tmp, M_peak, D,
H, tau for every batch from the configuration and compares with the manifest
fields (rel. tol 1e-9)."""


def _a(c, q):
    return c * q + q * (q + 1) // 2


def validate(inst, policy, records, X, kmax=1):
    errors = []
    m, dev = inst.model, inst.device
    kappa = 2 * m.L * m.n_kv * m.d_h * m.q_kv
    attn_w = m.n_q * m.d_h
    N, n = inst.N, inst.n
    z = [1] * N
    r = [0] * N
    pins = set()
    evaluated = set()          # (doc, stage) filter calls completed

    def err(t, msg):
        errors.append((t, msg))

    for rec in records:
        t = rec["t"]
        if rec["policy"] != policy:
            err(t, f"policy mismatch {rec['policy']} != {policy}")
        U = A = K_W = K_tmp = 0
        read_blocks = {}
        inbatch_doc = {}        # doc -> new doc tokens produced this batch
        inbatch_prompts = set()
        completions = []        # (doc, first_stage, k)
        branch_stages = {}      # doc -> sorted list of branch stages

        for op in rec["ops"]:
            kind, i, j = op["kind"], op["doc"], op["stage"]
            q = op["token_end"] - op["token_start"]
            if q <= 0:
                err(t, f"op with nonpositive tokens: {op}")
                continue
            if kind == "prompt_prefill":
                if policy != "task":
                    err(t, "prompt_prefill outside task-first")
                if j in pins:
                    err(t, f"prompt {j} already resident")
                if q != inst.p[j - 1]:
                    err(t, f"prompt {j} wrong length")
                inbatch_prompts.add(j)
                U += q; A += _a(0, q); K_W += q; K_tmp += q
            elif kind == "doc_chunk":
                if op["token_start"] != r[i]:
                    err(t, f"doc {i} chunk starts at {op['token_start']}, r={r[i]}")
                if op["token_end"] > inst.d[i]:
                    err(t, f"doc {i} chunk beyond d_i")
                if policy == "task":
                    if j != z[i]:
                        err(t, f"doc {i} chunk at stage {j}, frontier {z[i]}")
                    if z[i] > n:
                        err(t, f"doc {i} already finished")
                    if j in pins:
                        cache = inst.p[j - 1] + r[i]
                        read_blocks[("prompt", j)] = inst.p[j - 1]
                    elif j in inbatch_prompts:
                        cache = inst.p[j - 1] + r[i]
                    else:
                        err(t, f"doc {i} chunk without prompt {j} available")
                        cache = r[i]
                else:
                    if j != 0:
                        err(t, f"doc-first chunk with stage {j}")
                    cache = r[i]
                if r[i] > 0:
                    key = ("taskdoc", i) if policy == "task" else ("doc", i)
                    read_blocks[key] = r[i]
                U += q; A += _a(cache, q); K_W += q; K_tmp += q
                inbatch_doc[i] = inbatch_doc.get(i, 0) + q
                if r[i] + inbatch_doc[i] == inst.d[i] and policy == "task":
                    # write-through ledger: the final task-template position is
                    # the decision leaf with no descendant -> ephemeral
                    K_W -= 1
                    completions.append((i, z[i], 1))
            elif kind == "branch":
                if policy == "task":
                    err(t, "branch op under task-first")
                    continue
                if q != inst.p[j - 1]:
                    err(t, f"branch {i},{j} wrong prompt length")
                branch_stages.setdefault(i, []).append(j)
                if r[i] + inbatch_doc.get(i, 0) != inst.d[i]:
                    err(t, f"branch {i},{j} without complete document prefix")
                if r[i] > 0:
                    read_blocks[("doc", i)] = r[i]
                A += _a(inst.d[i], q)
                # write-through ledger: interior branch tokens are consumed by
                # later tokens of the same prompt and are stored; only the
                # final decision position is an ephemeral leaf
                U += q; K_W += q - 1; K_tmp += q
            else:
                err(t, f"unknown op kind {kind}")

        for i, stages in branch_stages.items():
            stages = sorted(stages)
            k = len(stages)
            if stages != list(range(z[i], z[i] + k)):
                err(t, f"doc {i} branches {stages} not contiguous from frontier {z[i]}")
            if policy == "pipe" and k != 1:
                err(t, f"pipeline lookahead {k} > 1 (speculation) on doc {i}")
            if policy == "spec" and k > kmax:
                err(t, f"lookahead {k} exceeds kmax {kmax}")
            if policy == "fullspec" and k != n - z[i] + 1:
                err(t, f"fullspec requires the whole remaining block on doc {i}")
            if z[i] > n:
                err(t, f"branches on finished doc {i}")
            completions.append((i, z[i], k))

        resident = sum(r) + sum(inst.p[j - 1] for j in pins)
        M_peak = m.W_mem + kappa * (resident + K_tmp)
        if M_peak + dev.S > dev.M:
            err(t, "peak memory exceeds device capacity")
        if inst.max_new_tokens is not None and U > inst.max_new_tokens:
            err(t, "new-token cap violated")
        K_L = sum(read_blocks.values())
        D = max(2.0 * m.P * U / dev.R_D, m.W_run / dev.BW) if U > 0 else 0.0
        H = max(4.0 * m.L * attn_w * A / dev.R_A, kappa * (K_L + K_W) / dev.BW)
        tau = D + H
        for name, got in (("U", U), ("A", A), ("K_L", K_L), ("K_W", K_W),
                          ("K_tmp", K_tmp)):
            if rec[name] != got:
                err(t, f"{name}: manifest {rec[name]} != recomputed {got}")
        for name, got in (("M_peak", M_peak), ("D", D), ("H", H), ("tau", tau)):
            if abs(rec[name] - got) > 1e-9 * max(1.0, abs(got)):
                err(t, f"{name}: manifest {rec[name]} != recomputed {got}")

        # apply chunk progress, then completions + reveals
        for i, q in inbatch_doc.items():
            r[i] += q
        pins |= inbatch_prompts
        for i, j0, k in completions:
            passes = 0
            for off in range(k):
                if not (1 <= j0 + off <= n):
                    err(t, f"filter ({i},{j0+off}) outside stage range")
                    break
                if (i, j0 + off) in evaluated:
                    err(t, f"filter ({i},{j0+off}) evaluated twice")
                evaluated.add((i, j0 + off))
                if X[i][j0 - 1 + off]:
                    passes += 1
                else:
                    break
            claimed = rec["outcomes"].get(str(i))
            if claimed is not None and claimed != passes:
                err(t, f"doc {i} outcome mismatch: manifest {claimed}, X gives {passes}")
            z[i] = j0 + k if (passes == k and j0 + k <= n) else n + 1
            if policy == "task":
                r[i] = 0
        for i in range(N):
            if z[i] == n + 1:
                r[i] = 0
        # evictions
        for i, keep in enumerate(rec["retained_r"]):
            if keep > r[i]:
                err(t, f"doc {i} retains {keep} > computed {r[i]}")
            r[i] = keep
        kept = set(rec["retained_pins"])
        if not kept.issubset(pins):
            err(t, "retained pins not a subset of resident pins")
        pins = kept

    # termination: every logically required evaluation done
    for i in range(N):
        if z[i] != n + 1:
            errors.append((None, f"doc {i} unfinished (z={z[i]})"))
        j = 1
        while j <= n:
            if (i, j) not in evaluated:
                errors.append((None, f"required filter ({i},{j}) never evaluated"))
                break
            if not X[i][j - 1]:
                break
            j += 1
    return errors
