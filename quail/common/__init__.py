"""Code the planner and the executor share.

The executor runs these at run time: token and pair tables, memory
budgets, answer-cost policies, and comparisons on columns. They import
only ``quail.specs``, so the executor can use them without importing
the planner.
"""
