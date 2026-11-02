"""PackSetPasses: the pass groups a set-attention plan prices and explains."""





def test_two_ai_filters_on_d1_2b_share_one_pass(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiFilter
    from quail.planner.plan import EngineConfig

    # documents much longer than a question, so one pass costs less
    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": [1, 2, 3], "body": [body] * 3}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('Is it kind? {0}', r.body)) "
        "AND AI_FILTER(PROMPT('Is it short? {0}', r.body))").plan()

    filters = [node for node in plan.nodes if isinstance(node, AiFilter)]
    assert [node.pass_group for node in filters] == [0, 0]
    assert all(node.state_tail_tokens == 1 for node in filters)
    assert all(node.question_header_token_ids == ("QUESTION:",) for node in filters)
    session.close()


def test_a_d1_plan_prices_and_explains_the_passes_its_operators_share(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    query = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), "
        "ARRAY['happy', 'sad']) AS mood, AI.CLASSIFY(PROMPT('Which size? {0}', "
        "r.body), ARRAY['short', 'long']) AS size FROM reviews r")

    plan = query.plan()
    first, second = [node.node_id for node in plan.nodes
                     if isinstance(node, AiClassify)]
    text = query.explain()

    assert plan.estimates[first]["pass_members"] == (first, second)
    assert plan.estimates[first]["seconds"] > 0
    assert "seconds" not in plan.estimates[second]
    assert f"Passes on r: [{first}, {second}]" in text
    assert "paged KV" not in text and "admission budget" not in text
    session.close()


def test_two_d1_classifications_share_a_pass_across_a_condition_between_them(
        tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('Which size? {0}', r.body), "
        "ARRAY['short', 'long']) AS size FROM reviews r WHERE "
        "AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), "
        "ARRAY['happy', 'sad']) = 'happy'").plan()

    labels = [node for node in plan.nodes if isinstance(node, AiClassify)]
    groups = {node.pass_group for node in labels}
    # the condition asks the model nothing, so both labels share a pass;
    # the executor applies the condition after it, and the second label
    # keeps only the rows the condition accepts
    assert len(labels) == 2 and len(groups) == 1 and -1 not in groups
    assert plan.estimates[labels[0].node_id]["pass_members"] == tuple(
        node.node_id for node in labels)
    session.close()


def test_a_d1_filter_and_classification_on_one_table_share_a_pass(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify, AiFilter
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    query = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), "
        "ARRAY['happy', 'sad']) AS mood FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Is it kind? {0}', r.body))")

    plan = query.plan()
    (kind,) = [node for node in plan.nodes if isinstance(node, AiFilter)]
    (label,) = [node for node in plan.nodes if isinstance(node, AiClassify)]

    # the filter keeps 20% of the 300-token reviews; asking the 4-token
    # mood question of every review costs less than reading the kept
    # reviews again
    assert kind.pass_group == label.pass_group >= 0
    assert plan.estimates[kind.node_id]["pass_members"] == (kind.node_id,
                                                           label.node_id)
    assert f"Passes on r: [{kind.node_id}, {label.node_id}]" in query.explain()
    session.close()


def test_a_d1_score_and_a_classification_of_one_table_share_a_pass(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify, AiScore
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    query = session.sql(
        "SELECT r.id, AI.SCORE(PROMPT('Is it funny? {0}', r.body)) AS fun, "
        "AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), ARRAY['happy', 'sad']) "
        "AS mood FROM reviews r ORDER BY fun DESC LIMIT 5")

    plan = query.plan()
    (score,) = [node for node in plan.nodes if isinstance(node, AiScore)]
    (label,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
    members = plan.estimates[score.node_id].get("pass_members") or \
        plan.estimates[label.node_id]["pass_members"]

    assert score.spec.state_tail_tokens == label.spec.state_tail_tokens > 0
    assert score.pass_group == label.pass_group >= 0
    assert sorted(members) == sorted((score.node_id, label.node_id))
    session.close()
