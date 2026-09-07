"""The QUAIL-B document sets: pinned sources, sampling, and identity.

Five document sets by default (IMDB, BioDEX, FEVER, LePaRD, SWE-Next
agent trace snapshots) plus the optional PrivacyPolicies set. Every
table is sampled from a pinned upstream revision with one seed, so a
scale factor names one exact corpus.
"""

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_SEED = 20260818
CACHE_SCHEMA_VERSION = 9
LEPARD_POSITIVE_PAIRS = 5_000
AGENT_TRACE_DOCUMENTS = 17_718
AGENT_TRACE_TURN_INTERVAL = 5
AGENT_TRACE_MAX_TOKENS = 24_000
AGENT_TRACE_TOKENIZER = "Qwen/Qwen3-4B-FP8"
AGENT_TRACE_TOKENIZER_REVISION = (
    "96b30dc13593a244a5e59e84687309f53c375cfa"
)

# Exact source snapshots for the benchmark corpus.  The row selection below
# is deterministic only when the upstream revisions are fixed as well as the
# sampling seed.
SOURCE_REVISIONS = {
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "BioDEX/BioDEX-Reactions":
        "01a5dacdabd144a120af04931a11a99febd48432",
    # FEVER's parquet files live on its conversion ref, so this is the
    # resolved commit for refs/convert/parquet rather than the default branch.
    "fever/fever": "5f577157472532aa1d9924d2df63aac44f70cf2b",
    "rmahari/LePaRD": "0194f95c3091acceab3b887c9b09ef432cf84052",
    "TIGER-Lab/SWE-Next-SFT-Trajectories":
        "e378a60ddd7050fe9519a31a4d41d4872eeec6ac",
    "mukund/PrivacyPolicies":
        "8fd6abfc7ca99d1f95c7f3f3a5dd5ea0cf9b7deb",
}

# Base document counts at sf=1. LePaRD scales sampled citation pairs
# before it deduplicates the two document tables.
SETS = {
    "reviews": 50_000,
    "reports": 5_000,
    "claims": 5_000,
    "agent_traces": AGENT_TRACE_DOCUMENTS,
    "policies": 1_000_000,
}

ASPECTS = ["the acting", "the plot", "the directing", "the cinematography",
           "the soundtrack", "the pacing", "the ending", "the dialogue",
           "the special effects", "the character development",
           "the screenplay", "the editing"]

SCENARIOS = [
    # marketing
    "You stop using the app, but months later you start getting ads "
    "from companies you have never heard of, based on things you "
    "searched for while you were still using it.",
    "You buy a product once, and then you keep getting emails and push "
    "notifications about similar products, even after you unsubscribe "
    "from the mailing list.",
    "You notice that the ads you see on other websites change right "
    "after you browse this service, as if your activity here followed "
    "you around the internet.",
    "You create an account just to try the free version, and within a "
    "week you start getting phone calls from salespeople who know your "
    "name and what features you looked at.",
    "You fill out a survey on the app, and later a completely different "
    "company contacts you about the exact topics you mentioned in "
    "your answers.",
    "You use the app for a few weeks, then delete it, but you keep "
    "seeing ads for it on social media that reference things you did "
    "inside the app.",
    "You sign up with a throwaway email, but the app somehow starts "
    "showing you ads related to purchases you made with your main "
    "email at other stores.",
    "You mention a product in a chat on the service, and within hours "
    "you see targeted ads for that exact product on other platforms.",
    "You opt out of marketing emails, but the company still sends you "
    "promotional messages disguised as account updates or security "
    "alerts.",
    "You notice the app suggesting friends who are customers of a "
    "partner company, even though you never shared your contacts.",
    # third-party sharing
    "A data broker contacts you with an offer, and when you ask how "
    "they got your information, they name this service as the source.",
    "You apply for a loan and the lender already has a profile of "
    "your spending habits, built from data this service shared with "
    "a financial analytics company.",
    "Your health insurance premium goes up, and when you investigate "
    "you find the insurer received wellness data that you entered "
    "into this app.",
    "A background check company has records of your activity on this "
    "service, even though you never gave them permission to access it.",
    "You discover that a political campaign has your personal details "
    "and browsing habits, traced back to a data-sharing agreement "
    "with this service.",
    "Your employer uses a workplace analytics tool that has data about "
    "your personal usage of this service, shared without your knowledge.",
    "A research firm publishes a study that includes aggregated data "
    "about users like you, and you can identify yourself from the "
    "details even though names were removed.",
    "You find your profile information listed on a people-search "
    "website, and the data matches exactly what you entered into "
    "this service.",
    "A retailer you have never visited sends you a coupon by mail, "
    "using your home address and product preferences from this app.",
    "You learn that a foreign government obtained your account data "
    "through a third party that this service shared it with.",
    # law enforcement
    "Police show up with a warrant for records of your activity on "
    "the service, and the company hands over six months of your chat "
    "history without telling you first.",
    "A government agency requests your location data from the past "
    "year, and the company provides it without requiring a court "
    "order.",
    "You are involved in a lawsuit, and the opposing side introduces "
    "your private messages from this service as evidence, obtained "
    "through a subpoena the company complied with.",
    "An immigration agency accesses your travel-related searches and "
    "account activity through a bulk data request to the company.",
    "You find out that the company gave law enforcement real-time "
    "access to your location for an investigation you were never "
    "told about.",
    "A tax authority receives your transaction records from this "
    "service as part of a compliance program the company participates "
    "in voluntarily.",
    "You are detained at a border crossing, and the officers already "
    "have a printout of your recent activity on this service.",
    "A local police department uses facial recognition to match a "
    "photo from your profile on this service to surveillance footage.",
    "Your account is flagged and frozen after the company runs an "
    "automated scan and reports your content to a government agency.",
    "A foreign court orders the company to hand over your data, and "
    "the company complies even though you live in a different country.",
    # data retention
    "You delete your account, but a year later you discover the "
    "company still has your photos stored on its servers.",
    "You request a copy of your data and find that the company kept "
    "records of searches you made five years ago, long after you "
    "stopped using the service.",
    "You close your account and later reopen one with the same email, "
    "and all your old preferences and history are still there.",
    "You ask the company to delete your data, they confirm it is "
    "done, but a data breach months later reveals your old records "
    "were still in their backup systems.",
    "You find out the company keeps a permanent record of every "
    "version of your profile, including photos and bios you changed "
    "years ago.",
    "You cancel your subscription, but the company continues to "
    "store and analyze your usage patterns for its own research.",
    "Your messages to other users remain visible to those users "
    "even after you delete your account, with your name still "
    "attached.",
    "You discover that the company retains your payment information "
    "indefinitely, even after you remove your credit card from the "
    "account settings.",
    "You move to a country with stricter data laws and request "
    "deletion, but the company says your data is stored in a "
    "jurisdiction where they are not required to delete it.",
    "You stop paying for the premium tier, but the company keeps "
    "all the data you uploaded during your subscription period "
    "without any stated expiration date.",
    # tracking
    "You use the app only at home, but it builds a detailed map of "
    "every store and restaurant you visit, using your phone's "
    "location in the background.",
    "You turn off location services for the app, but it still "
    "figures out where you are by scanning nearby Wi-Fi networks "
    "and Bluetooth devices.",
    "You browse the service on your laptop, and later when you open "
    "the app on your phone, it knows exactly which pages you visited "
    "on the laptop.",
    "You visit a physical store, and the app sends you a notification "
    "about a sale at that store moments later, even though you never "
    "searched for it.",
    "You notice the app has a record of how long you spend on each "
    "screen, how fast you scroll, and exactly where you tap.",
    "You clear your browser cookies, but the service still recognizes "
    "you the next time you visit, using device fingerprinting or "
    "other tracking methods.",
    "You use a VPN to hide your location, but the app still shows "
    "you local content, suggesting it has another way to determine "
    "where you are.",
    "You create a second account under a different name, but the "
    "service links it to your original account within days.",
    "You lend your phone to a friend, and the app records their "
    "usage pattern as yours, mixing their browsing into your profile.",
    "You find out the app tracks which other apps are installed on "
    "your phone and uses that information to build a profile of "
    "your interests.",
    # content and communications
    "You send a private photo to one person through the service, "
    "and later find it was scanned and flagged by the company's "
    "automated content review system.",
    "You write a private note in the app that you never share, and "
    "later the company uses the text to train a language model.",
    "You have a private video call on the service, and you later "
    "discover the company recorded and stored a transcript of the "
    "conversation.",
    "You upload a document to the service for personal storage, and "
    "the company uses its contents to improve its search algorithm.",
    "You send an encrypted message, but the company can still read "
    "it because the encryption keys are stored on the company's "
    "servers.",
    "You post something to a small private group, and the company's "
    "moderation system shares it with an external review team in "
    "another country.",
    "You draft a message but never send it, and later discover the "
    "company saved the draft and analyzed its contents.",
    "You share a voice message with a friend, and the company "
    "converts it to text and adds it to your advertising profile.",
    "You delete a post you made, but the company keeps a copy and "
    "continues to use it for content recommendations.",
    "You set your profile to private, but the company still allows "
    "search engines to index your profile photo and display name.",
    # AI and automated decisions
    "You apply for a service upgrade, and an algorithm denies your "
    "request based on your usage patterns, with no explanation and "
    "no way to appeal.",
    "The app automatically adjusts the prices you see based on how "
    "much it predicts you are willing to pay, without telling you.",
    "You are banned from the platform by an automated system that "
    "flagged your content, and no human ever reviews your appeal.",
    "The service uses your photos to train a facial recognition "
    "model, and that model is later sold to a company you have "
    "never interacted with.",
    "An algorithm decides which customer service tier you belong to, "
    "so your support tickets are deprioritized compared to users the "
    "system considers more valuable.",
    "You are shown a different version of the terms of service than "
    "other users, tailored by an algorithm based on your likelihood "
    "of reading the full text.",
    "The service uses your data to build a creditworthiness score "
    "that other companies can purchase and use in their own lending "
    "decisions.",
    "An automated system flags your account as suspicious based on "
    "your browsing patterns, and your access is restricted without "
    "any notification.",
    "The app uses your purchase history to predict your political "
    "views and sells that prediction to a data analytics firm.",
    "You receive different search results than other users because "
    "an algorithm decided what it thinks you want to see, without "
    "telling you it is personalizing.",
    # security and breaches
    "Your password is leaked in a data breach, and you find out "
    "about it from a news article before the company ever contacts "
    "you.",
    "The company suffers a breach that exposes your home address, "
    "phone number, and payment history, and offers you only one "
    "year of credit monitoring.",
    "Your biometric data, like a fingerprint or face scan, is stolen "
    "in a breach, and unlike a password, you cannot change it.",
    "You learn that an employee at the company accessed your account "
    "and read your private messages out of personal curiosity.",
    "The company stores your password in a way that allows anyone who "
    "breaks into their database to read it directly.",
    "A contractor working for the company downloads a database backup "
    "containing your data and takes it with them when they leave.",
    "Your account is taken over by someone who called the company's "
    "support line and convinced them to reset your password.",
    "The company shares your data with a partner whose security "
    "practices are weaker, and that partner gets breached.",
    "You discover that the company has no way to tell you which "
    "employees accessed your data or when.",
    "A security researcher publicly discloses a vulnerability that "
    "exposed your data, and the company had known about it for "
    "months without fixing it.",
    # children and family
    "Your thirteen-year-old child signs up for the service by "
    "entering a fake birth date, and the company collects and sells "
    "their data just like an adult's.",
    "You share a family account with your children, and the company "
    "builds advertising profiles for each family member, including "
    "the minors.",
    "Your child's school requires this service for homework, and "
    "the company uses the child's usage data for purposes beyond "
    "education.",
    "You give the app permission to access your contacts, and it "
    "starts sending messages to your children's phone numbers "
    "inviting them to join.",
    "You find out the company kept detailed records of your child's "
    "online activity from when they were ten years old, and those "
    "records are still accessible years later.",
    "The service recommends content to your teenager based on a "
    "profile built from data collected before they were old enough "
    "to consent.",
    "Your family's smart home device shares your children's voice "
    "recordings with this service, which uses them for product "
    "development.",
    "You set up parental controls, but the company's data collection "
    "practices apply the same way to your child's account as to "
    "yours.",
    "A classmate's parent uses the app to look up information about "
    "your child, and the service provides it because your child's "
    "profile is not fully private by default.",
    "You discover that the company used your child's data to train "
    "an AI model, even though your child's account was flagged as "
    "belonging to a minor.",
    # financial and sensitive data
    "You link your bank account to the service for payments, and the "
    "company uses your transaction history to build a spending profile "
    "that it shares with advertisers.",
    "You enter your Social Security number for identity verification, "
    "and the company stores it indefinitely, even after verification "
    "is complete.",
    "The service infers your income level from your usage patterns "
    "and uses it to decide which subscription plans to show you.",
    "You authorize a one-time payment, but the company stores your "
    "full credit card details and later charges you for a renewal "
    "you did not agree to.",
    "Your medical information, entered into a wellness feature of "
    "the app, is shared with an insurance company as part of a data "
    "partnership.",
    "The service tracks which financial articles you read and sells "
    "that behavioral data to investment firms.",
    "You discover that the company has been collecting information "
    "about your race, religion, or sexual orientation from your "
    "profile and activity, and using it for ad targeting.",
    "You apply for a job through the service, and the employer sees "
    "a risk score calculated from your financial data on the "
    "platform.",
    "You connect a fitness tracker to the app, and it shares your "
    "health metrics with third parties without a separate consent "
    "step.",
    "The service combines your purchase history with public records "
    "to estimate your net worth, and makes that estimate available "
    "to its business partners.",
]


def _n_docs(name, sf):
    return max(8, int(SETS[name] * sf))


def _n_lepard_pairs(sf):
    return max(8, int(LEPARD_POSITIVE_PAIRS * sf))


def _n_agent_documents(sf):
    return min(
        AGENT_TRACE_DOCUMENTS,
        max(8, round(AGENT_TRACE_DOCUMENTS * sf)),
    )


def _agent_message_text(message) -> str:
    """Render one agent message in the stored trace format."""
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(
            content, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False)
    return f"[{str(message.get('role', '')).upper()}]\n{content}"


def _agent_snapshot_boundaries(messages) -> tuple[str, list[tuple[int, int]]]:
    """Render one trace and return every fifth completed turn."""
    targets = {}
    assistant_turn = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn % AGENT_TRACE_TURN_INTERVAL:
            continue
        end = index + 1
        if end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        targets[end] = assistant_turn

    pieces = []
    boundaries = []
    length = 0
    for message_number, message in enumerate(messages, start=1):
        piece = _agent_message_text(message)
        if pieces:
            length += 2
        pieces.append(piece)
        length += len(piece)
        if message_number in targets:
            boundaries.append((targets[message_number], length))
    return "\n\n".join(pieces), boundaries


def _agent_has_issue(messages) -> bool:
    """Return whether the trace contains a nonempty user issue."""
    return any(
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def _agent_trace_rows(trace_index, messages, tokenizer) -> list[dict]:
    """Build the eligible snapshots for one SWE-Next trace."""
    if not _agent_has_issue(messages):
        return []
    text, boundaries = _agent_snapshot_boundaries(messages)
    rows = []
    for turn_index, end in boundaries:
        snapshot = text[:end]
        token_count = len(tokenizer.encode(
            snapshot, add_special_tokens=False))
        if token_count > AGENT_TRACE_MAX_TOKENS:
            continue
        rows.append({
            "id": f"at{trace_index:04d}-t{turn_index:03d}",
            "trace": snapshot,
            "trajectory_id": f"at{trace_index:04d}",
            "turn_index": turn_index,
            "token_count": token_count,
        })
    return rows


def _agent_rows(n):
    """Read SWE-Next and select a nested sample of trace snapshots."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = load_dataset(
        "TIGER-Lab/SWE-Next-SFT-Trajectories",
        split="train",
        revision=SOURCE_REVISIONS["TIGER-Lab/SWE-Next-SFT-Trajectories"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER,
        revision=AGENT_TRACE_TOKENIZER_REVISION,
    )
    order = np.random.default_rng(DATA_SEED).permutation(len(source))
    rows = []
    for trace_index in order:
        rows.extend(_agent_trace_rows(
            int(trace_index), source[int(trace_index)]["messages"],
            tokenizer))
        if len(rows) >= n:
            return rows[:n]
    raise ValueError(
        f"SWE-Next produced {len(rows)} eligible snapshots, expected {n}")


# ------------------------------------------------------- set builders

def _imdb_pool():
    from huggingface_hub import hf_hub_download
    texts = []
    for split in ("train", "test"):
        f = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["stanfordnlp/imdb"])
        texts += pq.read_table(f, columns=["text"]).column(
            "text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    return texts


def _biodex_rows(n):
    """Real BioDEX rows as (text, reactions), unpadded, un-concatenated.

    `reactions` seeds the `terms` table.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "BioDEX/BioDEX-Reactions", split="train", streaming=True,
        revision=SOURCE_REVISIONS["BioDEX/BioDEX-Reactions"])
    rows = []
    for row in ds:
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        reactions = [t.strip() for t in
                     str(row.get("reactions", "")).split(",") if t.strip()]
        if len(text) >= 200 and reactions:
            rows.append((text, reactions))
        if len(rows) >= n:
            break
    return rows


def _fever_data(n_claims):
    """FEVER claims (SUPPORTS/REFUTES only) and their Wikipedia pages.

    The evidence pool is bounded by the sampled claims.
    """
    from huggingface_hub import hf_hub_download
    seen, claims = set(), []
    for split in ("v1.0/train/0000.parquet",
                  "v1.0/labelled_dev/0000.parquet"):
        f = hf_hub_download("fever/fever", split,
                            repo_type="dataset",
                            revision=SOURCE_REVISIONS["fever/fever"])
        rows = pq.read_table(f).to_pylist()
        for r in rows:
            if (r["id"] in seen
                    or r["label"] not in ("SUPPORTS", "REFUTES")
                    or not r["evidence_wiki_url"]):
                continue
            seen.add(r["id"])
            claims.append(r)
            if len(claims) >= n_claims:
                break
        if len(claims) >= n_claims:
            break
    pages_needed = {r["evidence_wiki_url"] for r in claims}
    page_text = {}
    for shard in range(10):
        if len(page_text) >= len(pages_needed):
            break
        fw = hf_hub_download(
            "fever/fever",
            f"wiki_pages/partial-wikipedia_pages/{shard:04d}.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["fever/fever"])
        t = pq.read_table(fw, columns=["id", "text"])
        for pid, txt in zip(t.column("id").to_pylist(),
                            t.column("text").to_pylist()):
            if pid in pages_needed and pid not in page_text:
                page_text[pid] = txt
    return claims, page_text


def _lepard_pair_priority(dest_id, passage_id):
    value = f"{DATA_SEED}\0{dest_id}\0{passage_id}".encode()
    return int.from_bytes(
        hashlib.blake2b(value, digest_size=16).digest(), "big")


def _sample_lepard_pairs(rows, passages, n):
    """Select a stable random sample of distinct known citation pairs."""
    passages = {
        str(key): str(value).strip()
        for key, value in passages.items()
        if value
    }
    selected = []
    selected_rows = {}
    for dest_id, destination_context, passage_id in rows:
        dest_id = str(dest_id)
        passage_id = str(passage_id)
        key = (dest_id, passage_id)
        passage_text = passages.get(passage_id)
        context = str(destination_context).strip()
        if not passage_text or len(context) < 50:
            continue
        if key in selected_rows:
            prior_context, _ = selected_rows[key]
            if (len(context), context) > (len(prior_context), prior_context):
                selected_rows[key] = (context, passage_text)
            continue
        priority = _lepard_pair_priority(dest_id, passage_id)
        item = (-priority, dest_id, passage_id)
        if len(selected) < n:
            heapq.heappush(selected, item)
            selected_rows[key] = (context, passage_text)
            continue
        if priority >= -selected[0][0]:
            continue
        removed = heapq.heapreplace(selected, item)
        del selected_rows[(removed[1], removed[2])]
        selected_rows[key] = (context, passage_text)
    pairs = []
    for _, dest_id, passage_id in sorted(
            selected, key=lambda item: (-item[0], item[1], item[2])):
        context, passage_text = selected_rows[(dest_id, passage_id)]
        pairs.append((dest_id, passage_id, context, passage_text))
    return pairs


def _lepard_documents(pairs):
    """Deduplicate each document column after sampling citation pairs."""
    contexts = {}
    passages = {}
    for _dest_id, passage_id, context, passage_text in pairs:
        contexts.setdefault(context, set()).add(passage_id)
        passages.setdefault(passage_text, set()).add(passage_id)
    context_rows = [{
        "id": f"lc{i}",
        "destination_context": context,
        "cited_passage_ids": sorted(passage_ids),
    } for i, (context, passage_ids) in enumerate(contexts.items())]
    passage_rows = [{
        "id": f"lp{i}",
        "passage_text": passage_text,
        "passage_ids": sorted(passage_ids),
    } for i, (passage_text, passage_ids) in enumerate(passages.items())]
    return context_rows, passage_rows


def _lepard_rows(n):
    """Read LePaRD and sample known positive citation pairs."""
    import json as _json

    import pandas as pd
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download("rmahari/LePaRD", "top_10000_data.csv.gz",
                               repo_type="dataset",
                               revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    dict_path = hf_hub_download("rmahari/LePaRD", "passage_dict.json",
                                repo_type="dataset",
                                revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    with open(dict_path) as source:
        passages = _json.load(source)["data"]

    cols = ["dest_id", "destination_context", "passage_id"]
    chunks = pd.read_csv(
        csv_path,
        usecols=cols,
        chunksize=50_000,
        dtype={
            "dest_id": "string",
            "destination_context": "string",
            "passage_id": "string",
        },
    )
    rows = (
        row
        for chunk in chunks
        for row in chunk.loc[:, cols].itertuples(index=False, name=None)
    )
    return _sample_lepard_pairs(rows, passages, n)


def _vocab_table(rows, idx, cap=None):
    """A frequency-sorted, deduplicated vocabulary column from one field.

    Built across sampled rows (the `terms` table, from `reactions`).
    """
    freq = {}
    for row in rows:
        for t in row[idx]:
            freq[t] = freq.get(t, 0) + 1
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    return vocab[:cap] if cap else vocab


def _build_lepard(d, sf, force=False):
    """Build the two deduplicated LePaRD document tables."""
    context_path = d / "citation_contexts.parquet"
    passage_path = d / "citation_passages.parquet"
    if context_path.exists() and passage_path.exists() and not force:
        return
    expected_pairs = _n_lepard_pairs(sf)
    pairs = _lepard_rows(expected_pairs)
    if len(pairs) != expected_pairs:
        raise ValueError(
            f"LePaRD provided {len(pairs)} valid citation pairs, expected "
            f"{expected_pairs}")
    contexts, passages = _lepard_documents(pairs)
    context_schema = pa.schema([
        ("id", pa.string()),
        ("destination_context", pa.string()),
        ("cited_passage_ids", pa.list_(pa.string())),
    ])
    passage_schema = pa.schema([
        ("id", pa.string()),
        ("passage_text", pa.string()),
        ("passage_ids", pa.list_(pa.string())),
    ])
    pq.write_table(
        pa.Table.from_pylist(contexts, schema=context_schema), context_path)
    pq.write_table(
        pa.Table.from_pylist(passages, schema=passage_schema), passage_path)


def _build_policies(d, sf, force=False):
    """Build policies.parquet and scenarios.parquet, idempotent.

    The PrivacyPolicies corpus is ~1M documents and 48 GiB, so it may
    not be available on every machine. Skips silently when the source
    dataset is not installed.
    """
    pol_path = d / "policies.parquet"
    scen_path = d / "scenarios.parquet"
    if pol_path.exists() and scen_path.exists() and not force:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    n = _n_docs("policies", sf)
    try:
        f = hf_hub_download(
            "mukund/PrivacyPolicies",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["mukund/PrivacyPolicies"])
    except Exception:
        return
    t = pq.read_table(f, columns=["text"])
    texts = t.column("text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    texts = texts[:n]
    pq.write_table(pa.table({
        "id": [f"pp{i}" for i in range(len(texts))],
        "policy_text": texts,
    }), pol_path)
    pq.write_table(pa.table({
        "id": [f"sc{i}" for i in range(len(SCENARIOS))],
        "scenario": SCENARIOS,
    }), scen_path)


def _build_agent_traces(d, sf, force=False):
    """Build the SWE-Next cumulative trace snapshots."""
    path = d / "agent_traces.parquet"
    if path.exists() and not force:
        return
    rows = _agent_rows(_n_agent_documents(sf))
    schema = pa.schema([
        ("id", pa.string()),
        ("trace", pa.string()),
        ("trajectory_id", pa.string()),
        ("turn_index", pa.int32()),
        ("token_count", pa.int32()),
    ])
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=False,
    )


def build_sets(data_dir, sf, lf=1):
    """Build the benchmark tables as Parquet files, cached by sf.

    lf (load factor) is accepted but unused: documents here are real
    and unpadded, so there's nothing to scale. Kept in the signature
    so callers don't have to change when it's wired back up.
    """
    del lf
    d = Path(data_dir) / f"sf{sf}"
    marker = d / "DONE"
    if marker.exists():
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "data_seed": DATA_SEED,
            "lepard_positive_pairs": LEPARD_POSITIVE_PAIRS,
            "scale_factor": sf,
            "source_revisions": SOURCE_REVISIONS,
        }
        try:
            current = json.loads(marker.read_text())
        except json.JSONDecodeError:
            current = None
        if current == expected:
            _build_lepard(d, sf)
            _build_agent_traces(d, sf)
            return d
        base_sources = {
            name: revision for name, revision in SOURCE_REVISIONS.items()
            if name != "TIGER-Lab/SWE-Next-SFT-Trajectories"
        }
        same_sources = (
            current
            and current.get("data_seed") == DATA_SEED
            and current.get("scale_factor") == sf
            and all(
                current.get("source_revisions", {}).get(name) == revision
                for name, revision in base_sources.items()
            )
        )
        other_tables = (
            "reviews", "aspects", "reports", "terms", "claims", "evidence"
        )
        if same_sources and all((d / f"{name}.parquet").exists()
                                for name in other_tables):
            _build_lepard(d, sf, force=True)
            _build_agent_traces(d, sf, force=True)
            marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
            return d
    d.mkdir(parents=True, exist_ok=True)

    def write(name, ids, col_name, values):
        pq.write_table(pa.table({"id": ids, col_name: values}),
                       d / f"{name}.parquet")

    # reviews: real IMDB text, one row = one review, unpadded
    n = _n_docs("reviews", sf)
    imdb = _imdb_pool()
    write("reviews", [f"rv{i}" for i in range(n)], "body", imdb[:n])
    write("aspects", [f"as{i}" for i in range(len(ASPECTS))],
          "aspect", ASPECTS)

    # reports: real BioDEX text, one row = one report, unpadded
    n = _n_docs("reports", sf)
    bio = _biodex_rows(n)
    pq.write_table(pa.table({
        "id": [f"rp{i}" for i in range(len(bio))],
        "report": [t for t, _ in bio],
        "reactions": [r for _, r in bio],
    }), d / "reports.parquet")
    terms = _vocab_table(bio, 1)
    write("terms", [f"tm{i}" for i in range(len(terms))], "term", terms)
    # claims + evidence: real FEVER claims and only the Wikipedia
    # pages those claims reference
    n = _n_docs("claims", sf)
    claims, page_text = _fever_data(n)
    pq.write_table(pa.table({
        "id": [f"cl{i}" for i in range(len(claims))],
        "claim": [c["claim"] for c in claims],
        "label": [c["label"] for c in claims],
        "evidence_wiki_url": [c["evidence_wiki_url"] for c in claims],
    }), d / "claims.parquet")
    ev_ids = list(page_text.keys())
    pq.write_table(pa.table({
        "id": ev_ids,
        "text": [page_text[p] for p in ev_ids],
    }), d / "evidence.parquet")

    _build_lepard(d, sf, force=True)
    _build_agent_traces(d, sf, force=True)

    marker.write_text(json.dumps({
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "data_seed": DATA_SEED,
        "lepard_positive_pairs": LEPARD_POSITIVE_PAIRS,
        "scale_factor": sf,
        "source_revisions": SOURCE_REVISIONS,
    }, indent=2, sort_keys=True))
    return d


# ------------------------------------------------ corpus identity

# The columns of every table that take part in the corpus identity.
# The labeling pass hashes the same columns.
CORPUS_COLUMNS = {
    "reviews": ("id", "body"),
    "aspects": ("id", "aspect"),
    "reports": ("id", "report", "reactions"),
    "terms": ("id", "term"),
    "claims": ("id", "claim", "label", "evidence_wiki_url"),
    "evidence": ("id", "text"),
    "citation_contexts": ("id", "destination_context",
                          "cited_passage_ids"),
    "citation_passages": ("id", "passage_text", "passage_ids"),
    "agent_traces": ("id", "trace", "trajectory_id", "turn_index",
                     "token_count"),
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _full_hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _python_rows(rows) -> list[dict]:
    return rows.to_pylist() if isinstance(rows, pa.Table) else rows


def _row_id(rows, index: int):
    if isinstance(rows, pa.Table):
        return rows.column("id")[index].as_py()
    return rows[index]["id"]


def _ids(rows):
    if isinstance(rows, pa.Table):
        return rows.column("id").to_pylist()
    return [row["id"] for row in rows]


def corpus_identity(rows: dict[str, pa.Table | list[dict]],
                    scale_factor: float,
                    data_seed: int, source_revisions: dict) -> dict:
    tables = {}
    for table in sorted(rows):
        row_hashes = [_full_hash(row) for row in _python_rows(rows[table])]
        tables[table] = {
            "rows": len(row_hashes),
            "ordered_rows_full_hash": _full_hash(row_hashes),
        }
    payload = {
        "schema_version": 1,
        "benchmark": "quailb",
        "scale_factor": scale_factor,
        "data_seed": data_seed,
        "source_revisions": source_revisions,
        "tables": tables,
    }
    full = _full_hash(payload)
    return {
        **payload,
        "corpus_id": f"c_{full[:32]}",
        "corpus_full_hash": full,
    }


def read_corpus(data_dir: str | Path) -> dict[str, pa.Table]:
    data_dir = Path(data_dir)
    return {
        table: pq.read_table(
            data_dir / f"{table}.parquet", columns=list(columns))
        for table, columns in CORPUS_COLUMNS.items()
    }
