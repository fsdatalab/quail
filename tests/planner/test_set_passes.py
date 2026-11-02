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


def test_a_condition_between_two_d1_classifications_splits_their_passes(tmp_path):
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
    # the condition reads the first label, so the second is asked in a
    # later pass, as the executor runs them
    assert len(labels) == 2 and len(groups) == 2 and -1 not in groups
    assert all(plan.estimates[node.node_id]["pass_members"] == (node.node_id,)
               for node in labels)
    session.close()
