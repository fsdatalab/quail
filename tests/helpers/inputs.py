"""CPU input regressions: record projected provider reads and planned tokens."""

from helpers.oracle import Oracle, oracle_executor
from quail.api.execute import execute_query


class CountingProvider:
    """Delegate a provider while recording exactly which columns were requested."""

    def __init__(self, provider):
        self.provider = provider
        self.requests = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def scan(self, request):
        self.requests.append(request.columns)
        return self.provider.scan(request)


def snapshot_inputs(query):
    """Copy tokens and value columns before an executor consumes a region."""
    return {
        alias: {
            "tokens": scan.tokens.arrow_array.to_pylist(),
            "columns": {name: scan.column(name).to_pylist()
                        for name in scan.projected_columns},
        }
        for alias, scan in query.token_inputs().items()
    }


class RecordingRegions:
    """Run a true-answer oracle, preserving each region's actual planned inputs."""

    def __init__(self):
        self.inputs = []
        self.executor = oracle_executor(Oracle(
            filters=lambda alias, document, position: True,
            joins=lambda node, assignment: True,
        ))

    def __call__(self, query, on_rows, hint):
        self.inputs.append(snapshot_inputs(query))
        return execute_query(query, physical_executor=self.executor,
                             on_rows=on_rows, hint=hint).collect()
