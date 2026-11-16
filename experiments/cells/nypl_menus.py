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
CURRENCY_QUESTION = "Which currency are the prices on this menu in?"
LANGUAGE_QUESTION = "Which language are the dishes on this menu written in?"
EXPENSIVE_INSTRUCTION = ("Name the most expensive dish on this menu and its "
                         "price, as: dish | price. Write none if the menu shows "
                         "no prices.")
PRICE_INSTRUCTION = ("What does the menu charge for the dish named above? Answer "
                     "in dollars with a decimal point, such as 0.45 for 45 cents "
                     "or 1.25 for a dollar and a quarter.")
# the yes-or-no questions asked of each page image, answered per year
PAGE_QUESTIONS = {
    "alcohol": "Does this menu sell beer, wine, or liquor to drink? Dishes "
               "cooked with wine or liqueur do not count.",
    "war": "Does this menu mention a war, rationing, or a meatless or wheatless day?",
    "vegetarian": "Does this menu offer a vegetarian main course, with no meat, "
                  "poultry, or fish?",
}
# the prompts each page's image answers, all in one query so the model
# reads each image once: its key, the AI call, and the prompt text
def _ask(question: str) -> str:
    return question.replace("'", "''") + " {0}"


PAGE_CALLS = {
    **{key: f"AI.IF(PROMPT('{_ask(question)}', s.image))"
       for key, question in PAGE_QUESTIONS.items()},
    "currency": (f"AI.CLASSIFY(PROMPT('{_ask(CURRENCY_QUESTION)}', s.image), "
                 f"{CURRENCIES!r})"),
    "language": (f"AI.CLASSIFY(PROMPT('{_ask(LANGUAGE_QUESTION)}', s.image), "
                 f"{LANGUAGES!r})"),
    "most expensive dish": (f"AI.MAP(s.image, '{EXPENSIVE_INSTRUCTION}', 32)"),
}


def _pages_sql(keys: list[str], table: str) -> str:
    """One query asking each named page question of every page in the table."""
    return ("SELECT s.page_id, "
            + ", ".join(f"{PAGE_CALLS[key]} AS answer_{index}"
                        for index, key in enumerate(keys))
            + f" FROM {table} s")


PRICE_SQL = (f"SELECT s.page_id, AI.MAP(s.parts, '{PRICE_INSTRUCTION}', 8) AS answer "
             "FROM dollar_sample s")
SHRINK_SQL = "SELECT s.page_id, AI.IMAGE(s.image, {edge}) AS image FROM pages s"

# the query the docs page "A century of menus" shows: the share of each
# year's menus that sell drinks and that mention a war
MENU_HISTORY_SQL = f"""
WITH answers AS (
  SELECT p.menu_id,
         AI.IF(PROMPT('{_ask(PAGE_QUESTIONS["alcohol"])}', p.image)) AS sells_drinks,
         AI.IF(PROMPT('{_ask(PAGE_QUESTIONS["war"])}', p.image)) AS mentions_war
  FROM pages p)
SELECT m.year, COUNT(*) AS menus,
       ROUND(AVG(CASE WHEN a.sells_drinks THEN 1.0 ELSE 0.0 END), 3) AS drinks,
       ROUND(AVG(CASE WHEN a.mentions_war THEN 1.0 ELSE 0.0 END), 3) AS war
FROM answers a JOIN menus m ON a.menu_id = m.id
GROUP BY m.year HAVING COUNT(*) >= 20
ORDER BY m.year
"""


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes=volumes)
def prepare(num_menus: int, edge: int, seed: int, years: tuple[int, int] | None,
            per_year: int) -> dict:
    """Write the tables and one page image per menu to the volume.

    Each menu gives its page with the most typed-in dishes, or its first
    page when it has none, and one priced dish from that page when it has
    prices. With num_menus, half the menus have a typed-in currency and
    prices and half have neither; without it, every dated menu is kept.

    Args:
        num_menus: The menus in the sample, or 0 for every dated menu.
        edge: The page images' longest side, in pixels.
        seed: The sample's random seed.
        years: The first and last year of the menus to keep, or None for all.
        per_year: The most menus to keep from each year, or 0 for all.

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
    if years:
        candidates = [menu for menu in candidates
                      if years[0] <= year(menus[menu]) <= years[1]]
    if per_year:
        by_year: dict[int, list[str]] = {}
        for menu in candidates:
            by_year.setdefault(year(menus[menu]), []).append(menu)
        candidates = [menu for found in by_year.values()
                      for menu in chooser.sample(found, min(per_year, len(found)))]
    chosen = (chooser.sample(typed, num_menus // 2)
              + chooser.sample(untyped, num_menus - num_menus // 2)
              if num_menus else candidates)
    print(f"fetching {len(chosen)} pages", flush=True)

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
    rows = []
    with ThreadPoolExecutor(FETCH_THREADS) as pool:
        for index, row in enumerate(pool.map(fetch, chosen), start=1):
            if row:
                rows.append(row)
            if index % 200 == 0:
                print(f"fetched {index} of {len(chosen)} pages, {len(rows)} kept, "
                      f"{time.perf_counter() - started:.0f} s", flush=True)
    fetch_s = time.perf_counter() - started
    sample = pa.Table.from_pylist(rows)
    path = f"{ROOT}/{_sample_name(num_menus, edge, years, per_year)}"
    pq.write_table(sample, path)
    volumes[RESULTS].commit()
    return {"tables": counts, "menus": len(chosen), "sample": sample.num_rows,
            "fetch_s": round(fetch_s), "path": path}


def _sample_name(num_menus: int, edge: int, years: tuple[int, int] | None = None,
                 per_year: int = 0) -> str:
    span = f"_{years[0]}-{years[1]}" if years else ""
    cap = f"_{per_year}_per_year" if per_year else ""
    return f"pages_{num_menus or 'all'}{span}{cap}_{edge}.parquet"


def _prompt_tokens(session, prompt: str) -> int:
    """The tokens of a prompt around its document: its turn and template."""
    prefix, suffix = session.model.turn
    template = prompt.replace("''", "'").replace("{0}", "")
    return len(session.tokenizer(prefix + template + suffix))


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


def page_queries(session, sample, edge: int | None = None,
                 keys: list[str] | None = None) -> dict:
    """Ask every page question in one query, then read one price per dollar menu.

    Args:
        session: The open session.
        sample: The page sample: one row per menu with its page image.
        edge: The longest side to shrink each image to with AI.IMAGE, or
            None to read the images as stored.
        keys: The page questions to ask, or None for every question and
            the price query.

    Returns:
        The shrinking, the page query, and the price query, each with its
        measurements, answers, and scores.
    """
    import io

    import pyarrow as pa
    from PIL import Image

    import quail

    rows = sample.to_pylist()
    found = {"edge": edge}
    images = sample.select(["page_id", "image"])
    if edge:
        session.register(f"pages_{edge}", quail.DocumentProvider.from_table(
            images, id_col="page_id"))
        started = time.perf_counter()
        images = session.sql(SHRINK_SQL.format(edge=edge).replace(
            "FROM pages s", f"FROM pages_{edge} s")).collect().rename_columns(
                ["page_id", "image"])
        found["shrink_s"] = round(time.perf_counter() - started, 2)
    shrunk = dict(zip(images.column("page_id").to_pylist(),
                      images.column("image").to_pylist(), strict=True))
    sizes = {page: Image.open(io.BytesIO(image)).size for page, image in shrunk.items()}
    alias = f"sample_{edge or 'stored'}"
    session.register(alias, quail.DocumentProvider.from_table(
        images, id_col="page_id"))

    def tokens(page):
        width, height = sizes[page]
        return 2 + (width // 32) * (height // 32)

    by_page = {row["page_id"]: row for row in rows}
    asked = keys or list(PAGE_CALLS)
    prompts = [_prompt_tokens(session, re.search(
        r"PROMPT\('((?:[^']|'')*)'", PAGE_CALLS[key]).group(1))
        if "PROMPT(" in PAGE_CALLS[key] else
        _prompt_tokens(session, EXPENSIVE_INSTRUCTION) for key in asked]
    table, query_s = _timed(session, _pages_sql(asked, alias))
    input_tokens = sum(len(asked) * tokens(page) + sum(prompts) for page in shrunk)
    answers = {key: {} for key in asked}
    for row in _rows(table):
        for index, key in enumerate(asked):
            answers[key][row["page_id"]] = row[f"answer_{index}"]
    found["pages"] = {**_measured(query_s, input_tokens), "pages": len(shrunk),
                      "questions": asked, "sql": _pages_sql(asked, alias),
                      "image_tokens": sum(tokens(page) for page in shrunk),
                      "answers": answers}
    if "currency" in answers:
        known = {page: CURRENCY_NAMES[row["currency"]]
                 for page, row in by_page.items() if row["currency"] in CURRENCY_NAMES}
        found["pages"]["currency_accuracy"] = _share(answers["currency"], known)
    found["pages"]["years"] = {key: _shares_by_year(answers[key], by_page)
                               for key in PAGE_QUESTIONS if key in answers}
    print(f"pages at {edge or 'stored size'}: {query_s:.1f} s, "
          f"{found['pages']['input_tokens_per_second']:,} input tok/s, "
          f"currency {found['pages'].get('currency_accuracy')}", flush=True)
    if keys:
        return found

    dollar_rows = [row for row in rows if row["currency"] == "Dollars"
                   and row["dish"] and row["page_id"] in shrunk]
    priced = f"dollar_{alias}"
    session.register(priced, quail.DocumentProvider.from_table(pa.table({
        "page_id": [row["page_id"] for row in dollar_rows],
        "parts": pa.array([[shrunk[row["page_id"]], f"\n\nDish: {row['dish']}".encode()]
                           for row in dollar_rows], pa.list_(pa.binary())),
    }), id_col="page_id"))
    table, query_s = _timed(session, PRICE_SQL.replace("FROM dollar_sample s",
                                                       f"FROM {priced} s"))
    read = {row["page_id"]: row["answer"] for row in _rows(table)}
    input_tokens = sum(tokens(row["page_id"]) + len(session.tokenizer(
        f"\n\nDish: {row['dish']}")) for row in dollar_rows) + len(dollar_rows) * (
            _prompt_tokens(session, PRICE_INSTRUCTION))
    found["price"] = {
        **_measured(query_s, input_tokens), "pages": len(dollar_rows),
        "answers": read,
        "accuracy": _share({page: _number(answer) for page, answer in read.items()},
                           {row["page_id"]: row["price"] for row in dollar_rows},
                           close=True)}
    print(f"price at {edge or 'stored size'}: {query_s:.1f} s, "
          f"{found['price']['accuracy']}", flush=True)
    return found


def _shares_by_year(answers: dict, by_page: dict) -> dict:
    """The share of menus answering yes, and the menus asked, by year."""
    years: dict[int, list[bool]] = {}
    for page, row in by_page.items():
        if page in answers:
            years.setdefault(row["year"], []).append(bool(answers[page]))
    return {year: {"menus": len(said), "share": round(sum(said) / len(said), 3)}
            for year, said in sorted(years.items())}


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
def run(sample_path: str, run_id: str, read_edge: int) -> dict:
    """Run all three parts in one session; save the record to the volume.

    Args:
        sample_path: The page sample's parquet file on the volume.
        run_id: The run's folder on the volume.
        read_edge: The longest side the model reads each page at, or 0 for
            the stored size.
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
        record["pages"] = page_queries(session, pq.read_table(sample_path),
                                       read_edge or None)
    directory = f"{ROOT}/runs/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/record.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


@app.function(image=image, gpu="H100!", memory=98304, timeout=2 * 3600,
              volumes=volumes)
def compare_sizes(sample_path: str, edges: list[int], num_pages: int,
                  keys: list[str] | None, run_id: str) -> dict:
    """Ask the page questions at each image size; save the records to the volume.

    Args:
        sample_path: The page sample's parquet file on the volume.
        edges: The longest sides to shrink the pages to; 0 reads them as stored.
        num_pages: The sample's first pages to ask about, or 0 for all.
        keys: The page questions to ask, or None for every question and
            the price query.
        run_id: The run's folder on the volume.
    """
    import pyarrow.parquet as pq

    import quail

    sample = pq.read_table(sample_path)
    if num_pages:
        sample = sample.slice(0, num_pages)
    record = {"model": MODEL, "sample": sample_path, "sizes": {}}
    with quail.Session(quail.EngineConfig(model=MODEL, device="h100-sxm")) as session:
        for edge in edges:
            record["sizes"][str(edge)] = page_queries(session, sample, edge or None,
                                                      keys)
    directory = f"{ROOT}/runs/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/sizes.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600, volumes=volumes)
def history_query(sample_path: str, run_id: str) -> dict:
    """Run the docs page's query over the sample's pages; save its rows.

    Args:
        sample_path: The page sample's parquet file on the volume.
        run_id: The run's folder on the volume.

    Returns:
        The query's measurements and its rows.
    """
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    import quail

    sample = pq.read_table(sample_path)
    menus = sample.select(["menu_id", "year"]).rename_columns(["id", "year"])
    pages = sample.select(["page_id", "menu_id", "image"]).rename_columns(
        ["id", "menu_id", "image"])
    with quail.Session(quail.EngineConfig(model=MODEL, device="h100-sxm")) as session:
        session.register("menus", quail.DocumentProvider.from_table(
            pa.table(menus), id_col="id"))
        session.register("pages", quail.DocumentProvider.from_table(
            pages, id_col="id"))
        prompts = sum(_prompt_tokens(session, prompt) for prompt in re.findall(
            r"PROMPT\('((?:[^']|'')*)'", MENU_HISTORY_SQL))
        input_tokens = 0
        for image in pages.column("image").to_pylist():
            width, height = Image.open(io.BytesIO(image)).size
            input_tokens += 2 * (2 + (width // 32) * (height // 32)) + prompts
        table, query_s = _timed(session, MENU_HISTORY_SQL)
    record = {"model": MODEL, "sample": sample_path, "sql": MENU_HISTORY_SQL,
              "pages": pages.num_rows, **_measured(query_s, input_tokens),
              "rows": _rows(table)}
    directory = f"{ROOT}/runs/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/history.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


@app.local_entrypoint()
def history(sample: str = "pages_all_1914-1937_100_per_year_1024.parquet"):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{MODEL}-history"
    call = history_query.spawn(f"{ROOT}/{sample}", run_id)
    print(f"function call id: {call.object_id}", flush=True)
    record = call.get()
    print(f"saved to quail-results:/nypl_menus/runs/{run_id}/history.json", flush=True)
    print(f"{record['pages']} pages, {record['query_s']} s, "
          f"{record['input_tokens_per_second']:,} input tok/s, "
          f"${record['usd_per_query']}/query", flush=True)
    for row in record["rows"]:
        print(f"  {row['year']}  {row['menus']:>4}  "
              f"drinks {100 * row['drinks']:4.1f}%  war {100 * row['war']:4.1f}%")


@app.local_entrypoint()
def sizes(sample: str = "sample_2000_1024.parquet", edges: str = "512",
          num_pages: int = 0, questions: str = ""):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{MODEL}-sizes"
    call = compare_sizes.spawn(f"{ROOT}/{sample}",
                               [int(edge) for edge in edges.split(",")], num_pages,
                               questions.split(",") if questions else None, run_id)
    print(f"function call id: {call.object_id}", flush=True)
    record = call.get()
    print(f"saved to quail-results:/nypl_menus/runs/{run_id}/sizes.json", flush=True)
    for edge, found in record["sizes"].items():
        _summarize_pages(edge, found)


@app.local_entrypoint()
def prepare_data(num_menus: int = 0, edge: int = 1024, seed: int = 0,
                 years: str = "", per_year: int = 0):
    span = tuple(int(year) for year in years.split("-")) if years else None
    call = prepare.spawn(num_menus, edge, seed, span, per_year)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get()), flush=True)


@app.local_entrypoint()
def main(num_menus: int = 0, edge: int = 1024, read_edge: int = 0):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{MODEL}-{num_menus or 'all'}"
    call = run.spawn(f"{ROOT}/{_sample_name(num_menus, edge)}", run_id, read_edge)
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
    _summarize_pages(record["pages"]["edge"], record["pages"])


def _summarize_pages(edge, found: dict) -> None:
    """Print the page query's and the price query's measurements at one size."""
    pages, price = found["pages"], found.get("price")
    print(f"\npages at {edge or 'stored size'}: {pages['pages']} pages, "
          f"{pages['image_tokens']:,} image tokens, shrink {found.get('shrink_s')} s")
    for name, measured in [("page questions", pages), ("price", price)]:
        if measured:
            print(f"  {name:15} {measured['query_s']:7.1f} s "
                  f"{measured['input_tokens_per_second']:>9,} input tok/s "
                  f"${measured['usd_per_query']:.4f}/query")
    print(f"  currency {pages.get('currency_accuracy')}, "
          f"price {price['accuracy'] if price else None}")
    for key, years in pages["years"].items():
        print(f"  {key}: " + ", ".join(f"{year} {100 * found['share']:.0f}% "
                                      f"of {found['menus']}"
                                      for year, found in years.items()))
