"""Schedule manifest (paper eq. 57): serialization of a solved schedule so the
independent validator can replay it without solver internals."""

import json

from .costmodel import dense_time, attn_time
from .exact import engine
from .instance import Instance


def emit(inst: Instance, policy: str, schedule, X) -> list:
    """Convert an offline solver schedule into manifest records."""
    records = []
    for t, rec in enumerate(schedule, start=1):
        state, b = rec["state_before"], rec["batch"]
        ev = engine.evaluate_batch(inst, policy, state, b)
        assert ev is not None
        st, cost, ops = ev
        zs, rs, pins = state
        op_rows = []
        for op in ops:
            if op.kind == "doc_chunk":
                start = rs[op.doc]
                op_rows.append(dict(kind=op.kind, doc=op.doc, stage=op.stage,
                                    token_start=start, token_end=start + op.new))
            else:
                op_rows.append(dict(kind=op.kind, doc=op.doc, stage=op.stage,
                                    token_start=0, token_end=op.new))
        _zs_a, rs_a, pins_a = rec["state_after"]
        records.append(dict(
            t=t, policy=policy,
            ops=op_rows,
            outcomes={str(i): int(passes) for i, passes in rec["outcomes"].items()},
            retained_r=list(rs_a), retained_pins=sorted(pins_a),
            U=st.U, A=st.A, K_R=st.K_R, K_W=st.K_W, K_tmp=st.K_tmp,
            M_peak=inst.model.W_mem + inst.model.kappa
                   * (engine.resident_tokens(inst, state) + st.K_tmp),
            D=dense_time(inst.model, inst.device, st.U),
            H=attn_time(inst.model, inst.device, st.A, st.K_R, st.K_W),
            tau=cost,
        ))
    return records


def dump(records: list, path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
