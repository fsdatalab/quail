# Operator-at-a-time filter execution

The stock vLLM filter submission strategy is named `operator-at-a-time`.
The name uses standard database terminology in the configuration value,
execution function, tests, documentation, and reports.

Each filter still finishes for all remaining documents before the next filter
starts. This change only renames the strategy.
