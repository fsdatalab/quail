# Modal Function compute

`ModalComputeProvider` now calls a Modal Function in the existing
`quail-engine` app. Quail no longer starts an Arrow Flight server inside the
GPU container.

The provider sends the logical plan and source bindings to the function. A
remote source stays remote, so the function reads and tokenizes the documents
inside Modal. For a source that exists only in the client process, the provider
sends only the Arrow columns used by the query.

The worker returns the final Arrow table and execution report. The provider
prints the Modal function call id before it waits for the result. The provider
also keeps the selected function at one container while the session is open,
then allows it to scale to zero when the session closes.

Quail now has one built in compute provider. Benchmark functions that already
run on an H100 call the worker execution function directly.
