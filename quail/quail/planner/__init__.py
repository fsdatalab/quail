"""The planner: budgets from the specs, decisions from token
arithmetic plus one break-even inequality (read vs restore).
No wall prediction. KV is always bf16. a and a2 are loaded from
quail/calibration JSON; there is no measure step here."""
