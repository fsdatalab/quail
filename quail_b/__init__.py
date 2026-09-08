"""QUAIL-B: a benchmark of AI filter and join queries over document sets.

The package holds what defines the benchmark and nothing that runs it:
the dataset builders (`data`), the prompts (`prompts`), the query
specifications (`queries`), the predicates and the identity of their
labels (`predicates`), the saved reference labels (`labels`), the exact
prompt text a predicate asks (`rendering`), the file stores (`store`),
and the scoring of one run (`scoring`). A runner for one engine turns
the specifications into that engine's queries and hands the answers
back as a `RunOutput`.
"""
