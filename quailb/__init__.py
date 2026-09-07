"""QUAIL-B: a benchmark of AI filter and join queries over document sets.

The package holds what defines the benchmark and nothing that runs it:
the dataset builders (`data`), the prompts (`prompts`), the query
specifications (`queries`), the saved reference labels (`labels`), the
exact prompt text a predicate asks (`rendering`), the labeling pass
that produces the labels (`judge_pass`), and the scoring of one run
(`scoring`). A runner for one engine turns the specifications into
that engine's queries and hands the answers back as a `RunOutput`.
"""
