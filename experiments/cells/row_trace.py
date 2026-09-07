"""Observer used by the extension smoke cell; importable in the worker."""


class RowTrace:
    """Record each node's output rows as the worker finishes it."""

    name = "smoke.row_trace"

    def __init__(self):
        self.rows = []

    def after_node(self, node, result) -> None:
        self.rows.append([node.type_name, result.metrics.output_rows])

    def report(self) -> dict:
        return {"rows": list(self.rows)}
