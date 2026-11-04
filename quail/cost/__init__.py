"""Cost model: roofline arithmetic, operator prices, and retention prices.

Work counts and memory budgets, which the executor also uses, live in
``quail.common``.

Everything here is arithmetic over the model and device specs. Nothing
imports the planner or the runtime.
"""
