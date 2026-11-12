"""The three semantic top-k tasks: Sales, Customer Support, and Legal.

Each task is one QUAIL-B filter predicate over one table, read as a
ranking criterion. A record is relevant when Qwen3-32B's reference
label for that predicate is TRUE. Top-k accuracy is the share of the
returned k records that are relevant.
"""

import random
from dataclasses import dataclass

import pyarrow.compute as pc

SCALE_FACTOR = 0.1
PAIR_SEPARATOR = "\n\nRECORD 2:\n"
PAIR_HEAD = "RECORD 1:\n"


@dataclass(frozen=True)
class Task:
    """One top-k task over a QUAIL-B table.

    Attributes:
        name: Short task name used in result paths.
        query_id: A QUAIL-B query that loads the table and its labels.
        table: The QUAIL-B table.
        column: The text column each record is read from.
        predicate_key: The QUAIL-B predicate whose labels mark relevance.
        noun: What one record is, for the prompts.
        criterion: The ranking criterion, completing "whether ...".
        role: Keep only rows whose `role` column equals this, or None.
        k: How many records the top-k returns.
    """

    name: str
    query_id: str
    table: str
    column: str
    predicate_key: str
    noun: str
    criterion: str
    role: str | None
    k: int

    @property
    def pointwise_template(self) -> str:
        """The AI.SCORE prompt for one record, as QUAIL-B writes filters."""
        return (f"Judge strictly from the {self.noun} above whether "
                f"{self.criterion}\n\n{{0}}\n\nInstruction: answer TRUE if "
                f"{self.short_criterion}, FALSE otherwise.")

    @property
    def pair_template(self) -> str:
        """The AI.SCORE prompt for one pair: TRUE when RECORD 1 wins."""
        question = (f"Two {self.noun}s are shown above as RECORD 1 and "
                    f"RECORD 2. Judge strictly from them which one more "
                    f"clearly shows that {self.criterion}")
        return (question + "\n\n{0}\n\nInstruction: answer TRUE if RECORD 1 "
                "shows this more clearly than RECORD 2, FALSE otherwise.")

    @property
    def short_criterion(self) -> str:
        """The criterion's first clause, for the closing instruction."""
        return self.criterion.split(",")[0].rstrip(".")


TASKS = {
    "sales": Task(
        name="sales", query_id="SALES-5", table="sales_calls",
        column="transcript",
        predicate_key="quailb.sales.call.customer_commits",
        noun="sales call",
        criterion=("the customer commits to buying, such as accepting the "
                   "terms, agreeing to sign, or asking for the contract or "
                   "paperwork. Interest, questions, or agreeing to another "
                   "meeting do not count."),
        role=None, k=25),
    "support": Task(
        name="support", query_id="SUPPORT-1", table="support_messages",
        column="content",
        predicate_key="quailb.support.message.customer_frustrated",
        noun="customer message",
        criterion=("the customer expresses frustration or dissatisfaction "
                   "with the agent, such as repeating a request, objecting "
                   "to a refusal, or complaining about the service."),
        role="user", k=25),
    "legal": Task(
        name="legal", query_id="LEP-1", table="citation_contexts",
        column="destination_context",
        predicate_key="quailb.lepard.excerpt.reasoning_does_not_apply",
        noun="excerpt",
        criterion=("the excerpt argues that the cited case's reasoning does "
                   "not apply here."),
        role=None, k=15),
    "sales_outcome": Task(
        name="sales_outcome", query_id="SALES-5", table="sales_calls",
        column="transcript", predicate_key="",
        noun="sales call",
        criterion=("the customer is likely to sign a contract and buy, judging "
                   "by their commitment, urgency, and remaining objections."),
        role=None, k=25),
}


@dataclass(frozen=True)
class TaskData:
    """One task's records and labels.

    Attributes:
        task: The task.
        texts: Record id -> text.
        labels: Record id -> whether the record is relevant.
    """

    task: Task
    texts: dict[str, str]
    labels: dict[str, bool]

    @property
    def ids(self) -> list[str]:
        """Record ids in table order."""
        return list(self.texts)


def load_sales_outcomes() -> TaskData:
    """Each sampled deal's latest call, labeled by whether the deal was won.

    A deal is won when its CRMArena-Pro opportunity links to a contract
    the customer signed (status Activated, with a customer signed date).
    A won deal's record is its latest call dated before the signing
    date, so the call cannot report the signed contract; a won deal with
    no such call is left out. Records are keyed by QUAIL-B deal id, which
    follows quail_b.data._sales_rows: deals sorted by opportunity id,
    then permuted with DATA_SEED.
    """
    import sqlite3

    import numpy as np

    from quail_b import data as quailb_data

    texts, labels = {}, {}
    for domain in quailb_data.SALES_DOMAINS:
        path = quailb_data._download(quailb_data.CRMARENA_URL.format(
            commit=quailb_data.CRMARENA_COMMIT, domain=domain))
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            signed_on = dict(connection.execute(
                "SELECT o.Id, c.CustomerSignedDate FROM Opportunity o "
                "JOIN Contract c ON c.Id = o.ContractID__c "
                "WHERE c.Status = 'Activated' AND c.CustomerSignedDate IS NOT NULL"))
        finally:
            connection.close()
        calls = {}
        for deal, _, _, created, _, body in quailb_data._sales_source_calls(domain):
            calls.setdefault(deal, []).append((created, body))
        deals = sorted(calls)
        order = np.random.default_rng(quailb_data.DATA_SEED).permutation(len(deals))
        for index in order[:quailb_data._n_sales_deals(domain, SCALE_FACTOR)]:
            deal = deals[index]
            dated = sorted(calls[deal])
            if deal in signed_on:
                # ISO dates compare as strings; created times carry a time
                dated = [call for call in dated if call[0][:10] < signed_on[deal][:10]]
                if not dated:
                    continue
            key = f"sd{domain}{index:04d}"
            texts[key] = dated[-1][1]
            labels[key] = deal in signed_on
    return TaskData(TASKS["sales_outcome"], texts, labels)


def load_task(task: Task) -> TaskData:
    """Load a task's table and reference labels from the QUAIL-B bucket."""
    from quail_b.benchmark import load_benchmark

    if task.name == "sales_outcome":
        return load_sales_outcomes()

    benchmark = load_benchmark([task.query_id], scale_factor=SCALE_FACTOR)
    table = benchmark.tables[task.table]
    if task.role is not None:
        table = table.filter(pc.equal(table["role"], task.role))
    ids = [str(value) for value in table["id"].to_pylist()]
    texts = dict(zip(ids, table[task.column].to_pylist()))
    truth = benchmark.ground_truth
    labels = {key: bool(truth.answer(task.predicate_key, key)) for key in ids}
    return TaskData(task, texts, labels)


def pair_document(first: str, second: str) -> str:
    """The document of one comparison: RECORD 1, then RECORD 2."""
    return PAIR_HEAD + first + PAIR_SEPARATOR + second


def sample_swap_pairs(data: TaskData, num_pairs: int,
                      seed: int = 0) -> list[tuple[str, str]]:
    """Sample unordered pairs for the swap test: half mixed, half tied.

    Mixed pairs hold one relevant and one irrelevant record. Tied pairs
    hold two records with the same label, as many relevant-relevant
    pairs as exist up to a quarter of `num_pairs`, the rest
    irrelevant-irrelevant. Each pair is returned in a random order.
    """
    rng = random.Random(seed)
    positives = [key for key in data.ids if data.labels[key]]
    negatives = [key for key in data.ids if not data.labels[key]]
    pairs = set()

    def draw(left_pool, right_pool, count):
        attempts = 0
        start = len(pairs)
        while len(pairs) - start < count and attempts < count * 50:
            attempts += 1
            left, right = rng.choice(left_pool), rng.choice(right_pool)
            if left != right:
                pairs.add(tuple(sorted((left, right))))

    num_mixed = num_pairs // 2
    draw(positives, negatives, num_mixed)
    num_positive_ties = min(num_pairs // 4,
                            len(positives) * (len(positives) - 1) // 2)
    draw(positives, positives, num_positive_ties)
    draw(negatives, negatives, num_pairs - len(pairs))
    ordered = sorted(pairs)
    rng.shuffle(ordered)
    return [pair if rng.random() < 0.5 else pair[::-1] for pair in ordered]
