"""The execution layer: one entry point, backends chosen by the plan."""
from docengine.runtime.engine_client import (EngineTags,  # noqa: F401
                                             run_filter_chain,
                                             run_filter_chain_engine,
                                             run_query)


async def run_plan(engine, sampling_params, body_ids, q_ids, plan,
                   yes_ids=None, no_ids=None, tag="q"):
    """Execute a Plan on one worker's engine. Sharding across workers
    happens above (each worker receives its shard's body_ids); this
    dispatches the per-worker backend the plan chose."""
    if plan.mode == "chain":
        return await run_filter_chain_engine(
            engine, sampling_params, body_ids, q_ids,
            plan.budget_tokens, yes_ids, tag=tag,
            no_ids=no_ids if plan.stage_token_window > 1 else None)
    return await run_filter_chain(
        engine, sampling_params, body_ids, q_ids, plan.budget_tokens,
        lookahead=1, tag=tag,
        tags=EngineTags() if plan.pin else None, use_priority=True)
