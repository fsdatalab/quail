"""Implements paper.md's LP/exact-solver program: the exact reference
solvers, the expected-flow LP and its replay, the schedule manifest,
repricing, and the multi-GPU wrappers. This is the evidence behind the
first half of notes/RESULTS.md. Kept runnable for reproducibility, not
installed; the surviving package modules it builds on (costmodel,
instance, lb, sched.blockwise, validator) are imported from docengine.
"""
