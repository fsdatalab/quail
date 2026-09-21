"""Cost model: work counts, roofline arithmetic, budgets, and retention prices.

Everything here is arithmetic over the model and device specs. Nothing
imports the planner or the runtime. Image queries add the vision
tower's components beside the decoder's.
"""
