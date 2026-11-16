r"""Query the NYPL "What's on the menu?" collection, as ClickHouse does and further.

The New York Public Library's volunteers typed in the dishes and prices of
17,547 menus from the 1850s to the 2010s. The run asks three kinds of
question:

1. ClickHouse's demo queries, which read only the typed-in prices and
   find a dish with ILIKE on its name;
2. AI.IF on the dish names ILIKE matches for vodka and caviar, to check
   ILIKE's matches;
3. questions about one scanned page per menu: its currency and language;
   whether it offers alcohol, mentions a war, shows a price changed by
   hand, or offers a vegetarian main course; its most expensive dish; and
   a dish's price, checked against the volunteers' price.

    uv run modal run --detach -m experiments.cells.nypl_menus::prepare_data \
      2>&1 | tee "$run_log"
    uv run modal run --detach -m experiments.cells.nypl_menus::main \
      2>&1 | tee "$run_log"

prepare_data writes the tables and the page sample to the quail-results
volume under /nypl_menus/; the run saves its record under
/nypl_menus/runs/<run id>/record.json.
"""

from __future__ import annotations

import json
import os
import re
import time

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
RESULTS = "/results"
ROOT = f"{RESULTS}/nypl_menus"
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    RESULTS: modal.Volume.from_name("quail-results"),
}
H100_USD_PER_HOUR = 3.9492
MODEL = "qwen3.5-4b-bf16"
# the export ClickHouse's demo uses; its MD5 is db6126724de939a5481e3160a2d67d15
DATA_URL = ("https://s3.amazonaws.com/menusdata.nypl.org/gzips/"
            "2021_08_01_07_01_17_data.tgz")
PAGE_URL = "https://iiif.nypl.org/iiif/2/{image_id}/full/!{edge},{edge}/0/default.jpg"
# NYPL's image server answers in about a second but can hang a request
FETCH_TIMEOUT_S = 15
FETCH_ATTEMPTS = 6
FETCH_THREADS = 64
TABLES = {"menus": "Menu.csv", "pages": "MenuPage.csv", "items": "MenuItem.csv",
          "dishes": "Dish.csv"}
# the columns each table keeps, and which hold another table's id
COLUMNS = {
    "menus": ["id", "sponsor", "event", "place", "date", "currency"],
    "pages": ["id", "menu_id", "page_number", "image_id"],
    "items": ["id", "menu_page_id", "price", "dish_id"],
    "dishes": ["id", "name", "times_appeared"],
}
# every other column is a string, ids included, so joins compare strings
NUMBER_COLUMNS = {"price": "float64", "times_appeared": "int64",
                  "page_number": "int64"}

# ClickHouse's menu_item_denorm: one row per priced dish, with its menu's
# currency and date. ClickHouse rounds the year to the nearest decade.
PRICED_SQL = """
SELECT i.price, i.dish_id, d.name AS dish_name, m.currency AS menu_currency,
       CAST(ROUND(COALESCE(TRY_CAST(SUBSTR(m.date, 1, 4) AS INT), 0) / 10.0) * 10
            AS INT) AS decade
FROM items i
JOIN pages p ON i.menu_page_id = p.id
JOIN menus m ON p.menu_id = m.id
JOIN dishes d ON i.dish_id = d.id
"""

# ClickHouse's four demo queries, as ClickHouse wrote them, over PRICED_SQL
CLICKHOUSE_QUERIES = {
    "average price": ("menu_currency = 'Dollars'", ""),
    "burger": ("menu_currency = 'Dollars'", "dish_name ILIKE '%burger%'"),
    "vodka": ("menu_currency IN ('Dollars', '')", "dish_name ILIKE '%vodka%'"),
    "caviar": ("menu_currency IN ('Dollars', '')", "dish_name ILIKE '%caviar%'"),
}
# ClickHouse stores a missing price as 0, so its averages count each
# missing price as $0; the AI queries' averages skip missing prices
CLICKHOUSE_PRICE = "COALESCE(price, 0)"
DECADE_SQL = """
SELECT decade, COUNT(*) AS items, ROUND(AVG({price}), 2) AS average_price,
       MIN(dish_name) AS example
FROM priced
WHERE {currency} AND decade > 0 AND decade < 2022 {dish}
GROUP BY decade ORDER BY decade
"""

# each question's ILIKE pattern and the question AI.IF asks of each dish
# name ILIKE matches. ILIKE alone answers a burger, as in ClickHouse's demo.
DISH_QUESTIONS = {
    "vodka": ("%vodka%", "Is this drink vodka, or a drink made with vodka?"),
    "caviar": ("%caviar%", "Is this dish caviar, or served with caviar?"),
}
DISH_SQL = ("SELECT d.id FROM dishes d WHERE {condition}AI.IF(PROMPT('{question} "
            "Answer yes or no.\n\nDish: {{0}}', d.name))")

CURRENCIES = ["dollars", "francs", "marks", "pounds", "other currency", "no prices"]
# the volunteers' currency names, by the label that names the same money
CURRENCY_NAMES = {"Dollars": "dollars", "Francs": "francs",
                  "Deutsche Marks": "marks", "Pounds": "pounds",
                  "UK Pounds": "pounds"}
LANGUAGES = ["English", "French", "German", "Italian", "other"]
# the yes-or-no questions asked of each page image, answered per year
PAGE_QUESTIONS = {
    "alcohol": "Does this menu offer beer, wine, or liquor?",
    "war": "Does this menu mention a war, rationing, or a meatless or wheatless day?",
    "handwritten price": "Has someone crossed out a printed price on this menu "
                         "and written a new price by hand?",
    "vegetarian": "Does this menu offer a vegetarian main course, with no meat, "
                  "poultry, or fish?",
}
PAGE_QUERIES = {
    **{key: ("SELECT s.page_id, AI.IF(PROMPT('" + question.replace("'", "''")
             + " {0}', s.image)) AS answer FROM sample s")
       for key, question in PAGE_QUESTIONS.items()},
    "currency": (
        "SELECT s.page_id, AI.CLASSIFY(PROMPT('Which currency are the prices on "
        f"this menu in? {{0}}', s.image), {CURRENCIES!r}) AS answer FROM sample s"),
    "language": (
        "SELECT s.page_id, AI.CLASSIFY(PROMPT('Which language are the dishes on "
        f"this menu written in? {{0}}', s.image), {LANGUAGES!r}) AS answer "
        "FROM sample s"),
    "price": (
        "SELECT s.page_id, AI.MAP(s.parts, 'What does the menu charge for the "
        "dish named above? Answer in dollars, as a number only: 45 cents is "
        "0.45.', 8) AS answer FROM dollar_sample s"),
    "most expensive dish": (
        "SELECT s.page_id, AI.MAP(s.image, 'Name the most expensive dish on this "
        "menu and its price, as: dish | price. Write none if the menu shows no "
        "prices.', 32) AS answer FROM sample s"),
}


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes=volumes)
def prepare(num_menus: int, edge: int, seed: int) -> dict:
    """Write the tables and one page image per menu to the volume.

    Each menu gives its page with the most typed-in dishes, or its first
    page when it has none, and one priced dish from that page when it has
    prices. With num_menus, half the menus have a typed-in currency and
    prices and half have neither; without it, every dated menu is kept.

    Args:
        num_menus: The menus in the sample, or 0 for every dated menu.
        edge: The page images' longest side, in pixels.
        seed: The sample's random seed.

    Returns:
        The tables' row counts and the sample's size.
    """
    import io
    import random
    import tarfile
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    from PIL import Image

    os.makedirs(f"{ROOT}/tables", exist_ok=True)
    with urllib.request.urlopen(DATA_URL) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    tables, counts = {}, {}
    for name, file in TABLES.items():
        member = next(found for found in archive.getmembers()
                      if found.name.endswith(file))
        table = pcsv.read_csv(
            archive.extractfile(member),
            parse_options=pcsv.ParseOptions(newlines_in_values=True),
            convert_options=pcsv.ConvertOptions(
                include_columns=COLUMNS[name],
                column_types={column: NUMBER_COLUMNS.get(column, pa.string())
                              for column in COLUMNS[name]}))
        tables[name] = table
        counts[name] = table.num_rows
        pq.write_table(table, f"{ROOT}/tables/{name}.parquet")

    items = tables["items"]
    listed: dict[str, int] = {}
    for page in items.column("menu_page_id").to_pylist():
        listed[page] = listed.get(page, 0) + 1
    priced = items.filter(pc.greater(pc.fill_null(items.column("price"), 0.0), 0.0))
    by_page: dict[str, list[tuple[str, float]]] = {}
    for page, dish, price in zip(priced.column("menu_page_id").to_pylist(),
                                 priced.column("dish_id").to_pylist(),
                                 priced.column("price").to_pylist(), strict=True):
        by_page.setdefault(page, []).append((dish, price))
    dish_names = dict(zip(tables["dishes"].column("id").to_pylist(),
                          tables["dishes"].column("name").to_pylist(), strict=True))
    menus = {row["id"]: row for row in tables["menus"].to_pylist()}
    # each menu's page with the most typed-in dishes
    densest: dict[str, dict] = {}
    for row in tables["pages"].to_pylist():
        best = densest.get(row["menu_id"])
        key = (listed.get(row["id"], 0), -(row["page_number"] or 0))
        if best is None or key > (listed.get(best["id"], 0),
                                  -(best["page_number"] or 0)):
            densest[row["menu_id"]] = row

    def year(menu: dict) -> int | None:
        found = (menu.get("date") or "")[:4]
        return int(found) if found.isdigit() and 1850 <= int(found) <= 2021 else None

    # volunteers typed a currency almost only where they typed prices, so
    # half the sample has both and half has neither
    candidates = [menu for menu in densest if menu in menus and year(menus[menu])]
    typed = [menu for menu in candidates if menus[menu]["currency"]
             and by_page.get(densest[menu]["id"])]
    untyped = [menu for menu in candidates if not menus[menu]["currency"]
               and not by_page.get(densest[menu]["id"])]
    chooser = random.Random(seed)
    chosen = (chooser.sample(typed, num_menus // 2)
              + chooser.sample(untyped, num_menus - num_menus // 2)
              if num_menus else candidates)

    def fetch(menu: str) -> dict:
        page = densest[menu]
        url = PAGE_URL.format(image_id=page["image_id"], edge=edge)
        for attempt in range(FETCH_ATTEMPTS):
            try:
                with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
                    found = response.read()
                width, height = Image.open(io.BytesIO(found)).size
                break
            except OSError:
                time.sleep(min(2 ** attempt, 8))
        else:
            return {}
        dish, price = (random.Random(f"{seed}/{menu}").choice(by_page[page["id"]])
                       if page["id"] in by_page else (None, None))
        return {"page_id": page["id"], "menu_id": menu,
                "year": year(menus[menu]), "currency": menus[menu]["currency"],
                "sponsor": menus[menu]["sponsor"], "event": menus[menu]["event"],
                "place": menus[menu]["place"], "image": found, "width": width,
                "height": height, "dish": dish_names.get(dish) or "",
                "price": price}

    started = time.perf_counter()
    with ThreadPoolExecutor(FETCH_THREADS) as pool:
        rows = [row for row in pool.map(fetch, chosen) if row]
    fetch_s = time.perf_counter() - started
    sample = pa.Table.from_pylist(rows)
    path = f"{ROOT}/{_sample_name(num_menus, edge)}"
    pq.write_table(sample, path)
    volumes[RESULTS].commit()
    return {"tables": counts, "menus": len(chosen), "sample": sample.num_rows,
            "fetch_s": round(fetch_s), "path": path}


def _sample_name(num_menus: int, edge: int) -> str:
    return f"pages_{num_menus or 'all'}_{edge}.parquet"


def _prompt_tokens(session, prompt: str) -> int:
    """The tokens of a prompt around its document: its turn and template."""
    prefix, suffix = session.model.turn
    template = prompt.replace("''", "'").replace("{0}", "")
    return len(session.tokenizer(prefix + template + suffix))


def _image_tokens(row: dict) -> int:
    return 2 + (row["width"] // 32) * (row["height"] // 32)


def _timed(session, sql: str):
    """Run an AI query; return its table and its query time in seconds."""
    from bench.quailb import _submission_to_answer_s

    submitted = time.perf_counter()
    query = session.sql(sql)
    frontend_s = time.perf_counter() - submitted
    result = query.run()
    query_s = _submission_to_answer_s(
        "quail", result.report, frontend_s=frontend_s, answer_prepare_s=0.0)
    return result.collect(), query_s


def _measured(query_s: float, input_tokens: int) -> dict:
    return {"query_s": round(query_s, 2), "input_tokens": input_tokens,
            "input_tokens_per_second": round(input_tokens / query_s),
            "usd_per_query": round(query_s / 3600 * H100_USD_PER_HOUR, 4)}


def _rows(table) -> list[dict]:
    return [{name.split(".")[-1]: value for name, value in row.items()}
            for row in table.to_pylist()]


def clickhouse_queries(session) -> dict:
    """Run ClickHouse's demo queries; return each one's rows and seconds."""
    found = {}
    for name, (currency, dish) in CLICKHOUSE_QUERIES.items():
        started = time.perf_counter()
        table = session.sql(DECADE_SQL.format(
            price=CLICKHOUSE_PRICE, currency=currency,
            dish=f"AND {dish}" if dish else "")).collect()
        found[name] = {"seconds": round(time.perf_counter() - started, 2),
                       "rows": _rows(table)}
        print(f"clickhouse {name}: {table.num_rows} decades", flush=True)
    return found


def dish_queries(session, dishes) -> dict:
    """Ask each dish question with AI.IF; compare its dishes with ILIKE's.

    Returns:
        For each question: the query's measurements, the dishes ILIKE and
        AI.IF each find, the dishes only one finds, and both decade tables.
    """
    import pyarrow as pa

    import quail

    names = dict(zip(dishes.column("id").to_pylist(),
                     dishes.column("name").to_pylist(), strict=True))
    times = dict(zip(dishes.column("id").to_pylist(),
                     dishes.column("times_appeared").to_pylist(), strict=True))

    def top(ids):
        return [names[dish] for dish in sorted(
            ids, key=lambda dish: -(times[dish] or 0))[:25]]

    found = {}
    for key, (pattern, question) in DISH_QUESTIONS.items():
        sql = DISH_SQL.format(condition=f"d.name ILIKE '{pattern}' AND ",
                              question=question.replace("'", "''"))
        prompt = re.search(r"PROMPT\('((?:[^']|'')*)'", sql).group(1)
        table, query_s = _timed(session, sql)
        hits = {row["id"] for row in _rows(table)}
        needle = pattern.strip("%").lower()
        matched = {dish for dish, name in names.items() if needle in name.lower()}
        asked = matched
        input_tokens = sum(len(session.tokenizer(names[dish])) for dish in asked) + (
            len(asked) * _prompt_tokens(session, prompt))
        alias = f"hits_{key}"
        session.register(alias, quail.DocumentProvider.from_table(
            pa.table({"id": sorted(hits)}), id_col="id"))
        decades = {}
        for method, condition in [
                ("ilike", f"dish_name ILIKE '{pattern}'"),
                ("ai", f"dish_id IN (SELECT id FROM {alias})")]:
            decades[method] = _rows(session.sql(DECADE_SQL.format(
                price="price", currency="menu_currency IN ('Dollars', '')",
                dish=f"AND {condition} AND price > 0")).collect())
        found[key] = {**_measured(query_s, input_tokens), "sql": sql,
                      "asked": len(asked), "ilike": len(matched),
                      "ai": len(hits), "both": len(matched & hits),
                      "ai_only": top(hits - matched),
                      "ilike_only": top(matched - hits), "decades": decades}
        print(f"dish {key}: asked {len(asked)}, ILIKE {len(matched)}, "
              f"AI.IF {len(hits)}, {query_s:.1f} s", flush=True)
    return found


def page_queries(session, sample) -> dict:
    """Ask each page question of the sampled page images; score what has labels."""
    import pyarrow as pa

    import quail

    rows = sample.to_pylist()
    session.register("sample", quail.DocumentProvider.from_table(
        sample.select(["page_id", "image"]), id_col="page_id"))
    dollar_rows = [row for row in rows if row["currency"] == "Dollars"]
    session.register("dollar_sample", quail.DocumentProvider.from_table(pa.table({
        "page_id": [row["page_id"] for row in dollar_rows],
        "parts": pa.array([[row["image"], f"\n\nDish: {row['dish']}".encode()]
                           for row in dollar_rows], pa.list_(pa.binary())),
    }), id_col="page_id"))
    image_tokens = sum(_image_tokens(row) for row in rows)
    by_page = {row["page_id"]: row for row in rows}
    found = {}
    for key, sql in PAGE_QUERIES.items():
        table, query_s = _timed(session, sql)
        answers = {row["page_id"]: row["answer"] for row in _rows(table)}
        if key == "most expensive dish":
            instruction = re.search(r"AI.MAP\(s.image, '((?:[^']|'')*)'", sql).group(1)
            input_tokens = image_tokens + len(rows) * _prompt_tokens(
                session, instruction)
        elif key == "price":
            instruction = re.search(r"AI.MAP\(s.parts, '((?:[^']|'')*)'", sql).group(1)
            input_tokens = sum(
                _image_tokens(row)
                + len(session.tokenizer(f"\n\nDish: {row['dish']}"))
                for row in dollar_rows) + len(dollar_rows) * _prompt_tokens(
                    session, instruction)
        else:
            prompt = re.search(r"PROMPT\('((?:[^']|'')*)'", sql).group(1)
            input_tokens = image_tokens + len(rows) * _prompt_tokens(session, prompt)
        record = {**_measured(query_s, input_tokens), "pages": len(answers),
                  "answers": answers}
        if key == "currency":
            known = {page: CURRENCY_NAMES[row["currency"]]
                     for page, row in by_page.items()
                     if row["currency"] in CURRENCY_NAMES}
            record["accuracy"] = _share(answers, known)
            # the menus with no typed-in prices, by what the model reads
            untyped = [answers.get(page) for page, row in by_page.items()
                       if not row["currency"]]
            record["untyped"] = {label: untyped.count(label) for label in CURRENCIES}
        elif key in PAGE_QUESTIONS:
            # the share of menus answering yes, by year
            years: dict[int, list[bool]] = {}
            for page, row in by_page.items():
                years.setdefault(row["year"], []).append(bool(answers.get(page)))
            record["years"] = {year: {"menus": len(said),
                                      "share": round(sum(said) / len(said), 3)}
                               for year, said in sorted(years.items())}
        elif key == "price":
            known = {page: row["price"] for page, row in by_page.items()
                     if row["currency"] == "Dollars"}
            record["accuracy"] = _share(
                {page: _number(answer) for page, answer in answers.items()},
                known, close=True)
        found[key] = record
        print(f"page {key}: {query_s:.1f} s, {record.get('accuracy')}", flush=True)
    return found


def _number(text: str | None) -> float | None:
    found = re.search(r"\d+(?:\.\d+)?", (text or "").replace(",", ""))
    return float(found.group()) if found else None


def _share(answers: dict, known: dict, close: bool = False) -> dict:
    """The share of labeled pages whose answer matches the label."""
    def right(answer, label):
        if close:
            return answer is not None and abs(answer - label) < 0.005
        return answer == label

    scored = [right(answers.get(page), label) for page, label in known.items()]
    return {"accuracy": round(sum(scored) / max(1, len(scored)), 3),
            "labeled": len(scored)}


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=volumes)
def run(sample_path: str, run_id: str) -> dict:
    """Run all three parts in one session; save the record to the volume.

    Args:
        sample_path: The page sample's parquet file on the volume.
        run_id: The run's folder on the volume.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    import quail

    tables = {name: pq.read_table(f"{ROOT}/tables/{name}.parquet")
              for name in TABLES}
    dishes = tables["dishes"]
    dishes = dishes.filter(pc.and_(
        pc.greater(pc.fill_null(dishes.column("times_appeared"), 0), 0),
        pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(
            pc.fill_null(dishes.column("name"), ""))), 0)))
    record = {"model": MODEL, "sample": sample_path}
    config = quail.EngineConfig(model=MODEL, device="h100-sxm")
    with quail.Session(config) as session:
        for name, table in tables.items():
            if name == "dishes":
                table = dishes
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col="id"))
        priced = session.sql(PRICED_SQL).collect().rename_columns(
            ["price", "dish_id", "dish_name", "menu_currency", "decade"])
        priced = priced.append_column(
            "id", pa.array([str(index) for index in range(priced.num_rows)]))
        session.register("priced", quail.DocumentProvider.from_table(
            priced, id_col="id"))
        record["clickhouse"] = clickhouse_queries(session)
        record["dishes"] = dish_queries(session, dishes)
        record["pages"] = page_queries(session, pq.read_table(sample_path))
    directory = f"{ROOT}/runs/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/record.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


@app.local_entrypoint()
def prepare_data(num_menus: int = 0, edge: int = 1024, seed: int = 0):
    call = prepare.spawn(num_menus, edge, seed)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get()), flush=True)


@app.local_entrypoint()
def main(num_menus: int = 0, edge: int = 1024):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{MODEL}-{num_menus or 'all'}"
    call = run.spawn(f"{ROOT}/{_sample_name(num_menus, edge)}", run_id)
    print(f"function call id: {call.object_id}", flush=True)
    record = call.get()
    print(f"saved to quail-results:/nypl_menus/runs/{run_id}/record.json", flush=True)
    summarize(record)


def summarize(record: dict) -> None:
    """Print each part's results: ClickHouse's tables, then Quail's queries."""
    for name, found in record["clickhouse"].items():
        print(f"\nClickHouse {name} ({found['seconds']} s)")
        for row in found["rows"]:
            print(f"  {row['decade']}  {row['items']:>7}  ${row['average_price']}")
    for key, found in record["dishes"].items():
        print(f"\n{key}: asked {found['asked']} dishes; ILIKE finds {found['ilike']}, "
              f"AI.IF {found['ai']}, "
              f"both {found['both']}; {found['query_s']} s, "
              f"{found['input_tokens_per_second']:,} input tok/s, "
              f"${found['usd_per_query']}/query")
        print(f"  AI.IF only: {found['ai_only'][:8]}")
        print(f"  ILIKE only: {found['ilike_only'][:8]}")
    for key, found in record["pages"].items():
        print(f"\npage {key}: {found['pages']} pages, {found['query_s']} s, "
              f"{found['input_tokens_per_second']:,} input tok/s, "
              f"${found['usd_per_query']}/query, {found.get('accuracy')}")
        for name in ("untyped",):
            if name in found:
                print(f"  {name}: {found[name]}")
