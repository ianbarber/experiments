"""LongMemEval_S (cleaned) as file-based memory stores, for the hint-vs-access check (PREREG §H1).

A question's history becomes a store of files, one per session, named by date and time
("2023-05-20_02-21.md"). Sessions with a repeated id are dropped (first kept) and the rest are
stably sorted by date, so the store and the compiled memory are in date order. Stores:
  present  the history
  absent   every evidence session (answer_session_ids) replaced, in place and under the same file
           name, by a personal-style session from another history (simulated user chat, not an
           evidence session): characters within ±15% (uniformly among the 20 closest), not already
           in this history, not identical to an evidence session of this question or its _abs
           twin, sharing no (stemmed) content word of the question in its user turns, and not
           stating the gold answer in its user turns (word-boundary match; purely numeric and
           yes/no golds exempt, since counts are never stated and digits are everywhere)
  absent2  the same rule with an independent draw that excludes absent's picks (placebo swap)
The compiled memory is the store's files concatenated in order, so present and both absents share
every token before the first evidence file.

Also: a keyword index of a store (top TF-IDF words per file, IDF fixed from every session in the
dataset, so present and absent indexes differ only on evidence lines), BM25 over a store's files
(where a generated search query points), the tool schema, the sample, the outcome-blind audit of
absent stores, and the exclusion list.
"""
import json
import math
import random
import re
from collections import Counter

STOP = set("""a about above after again against all also am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom
why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
will just like also get got make made really much many well way want need know think thing things good great
one two three first new use used using sure help may might yes okay thanks thank please here's lot""".split())
# generic request / framing words: not topic-bearing, so they do not exclude replacement sessions
GENERIC = set("""recommend recommendations suggest suggestions suggestion tips ideas idea advice give tell
find best some more learn current currently recently recent last ago time times day days week weeks month
months year years today tonight tomorrow anything something everything someone people person able
thinking trying planning looking getting going start started starting total initially kind type types
different several ever previous previously remember mentioned told said ones""".split())
WORD = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")
NUMERIC = re.compile(r"^[\d$.,%\- ]+$")

NEUTRAL = ("You are a helpful assistant. You have memory files from past conversations with this user "
           "(one file per session, named by date and time). Use the memory tools when something the user "
           "told you before could matter.")
TOOLS = [
    {"type": "function", "function": {"name": "list_memory", "description": "List the memory files (one per past session, named by date and time).",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "search_memory", "description": "Search all memory files for a word or phrase. Returns matching lines with their file names.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Word or phrase to search for."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_memory", "description": "Read one memory file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File name, e.g. 2023-05-20_02-21.md"}}, "required": ["path"]}}},
]

COUNTS = {"single-session-user": 24, "single-session-preference": 24, "knowledge-update": 16, "multi-session": 16}
# Outcome-blind exclusions, fixed before the run (rules in PREREG §H1; reasons logged in LABNOTES).
EXCLUDE = {
    # LongMemEval issues (snapshot 2026-10-08) documenting that the evidence does not support the gold
    "a3332713": "issue #11: evidence gives a different total than the gold",
    "dd2973ad": "issue #21: the bedtime in the evidence is not the day before the appointment",
    "618f13b2": "issue #38: a later session contradicts the gold count",
    "37f165cf": "issue #39: neither novel was finished in January or March",
    # audit: a kept non-evidence session answers the question in part or states the rubric preference
    "d851d5ba": "kept session d77d4ac9_1 states money raised for charity",
    "1c0ddc50": "kept session 2aa70c9c_1 states listening to podcasts on the bus commute",
    "75832dbd": "kept session sharegpt_xni9yYO_6 states wanting to publish in AI conferences",
    "35a27287": "kept session 880ee13e states taking language exchange classes",
    "b6025781": "kept session (2023-05-24 17:44) asks for healthy recipes for meal planning",
}


def words(text):
    return WORD.findall(text.lower())


def stem(w):
    w = w[:-2] if w.endswith("'s") else w
    if w.endswith("ies") and len(w) >= 6:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith("ss") and len(w) >= 5:
        w = w[:-1]
    if w.endswith("ing") and len(w) >= 7:
        w = w[:-3]
    elif w.endswith("ed") and len(w) >= 6:
        w = w[:-2]
    return w


def content_stems(question):
    return sorted({stem(w) for w in words(question)
                   if len(w) >= 3 and w not in STOP and w not in GENERIC and stem(w) not in GENERIC})


def user_text(sess):
    return " ".join(t["content"] for t in sess if t["role"] == "user")


def parse_date(d):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", d)
    return tuple(int(g) for g in m.groups())


def file_names(dates):
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20_02-21.md', de-duplicated with -2, -3 ..."""
    out, seen = [], Counter()
    for d in dates:
        y, mo, da, h, mi = parse_date(d)
        base = f"{y:04d}-{mo:02d}-{da:02d}_{h:02d}-{mi:02d}"
        seen[base] += 1
        out.append(f"{base}.md" if seen[base] == 1 else f"{base}-{seen[base]}.md")
    return out


def session_body(sess):
    return "\n".join(f"[{t['role']}]: {t['content'].strip()}" for t in sess)


def history(x):
    """[(session id, date, session)]: repeated ids dropped (first kept), stably sorted by date."""
    seen, out = set(), []
    for sid, d, s in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
        if sid not in seen:
            seen.add(sid)
            out.append((sid, d, s))
    return sorted(out, key=lambda r: parse_date(r[1]))


def is_personal(sid):
    return not (sid.startswith("sharegpt") or sid.startswith("ultrachat") or sid.startswith("answer_"))


def load(path):
    return json.load(open(path))


def build_pool(data):
    """Personal-style filler sessions (simulated user chats that are not evidence), unique by id:
    sid -> (body, stems of user-turn words, lower-cased user text)."""
    pool = {}
    for x in data:
        for sid, s in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            if is_personal(sid) and sid not in pool:
                u = user_text(s).lower()
                pool[sid] = (session_body(s), {stem(w) for w in words(u) if len(w) >= 3}, u)
    return pool


def build_idf(data):
    """IDF over every unique session in the dataset (fixed reference for the keyword index)."""
    df, n = Counter(), 0
    seen = set()
    for x in data:
        for sid, s in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            if sid in seen:
                continue
            seen.add(sid)
            n += 1
            df.update(set(words(session_body(s))))
    return {w: math.log(n / c) for w, c in df.items()}, math.log(n)


def gold_patterns(x):
    """Word-boundary patterns for the gold answer (full if <= 60 chars, and cut at ' ('); purely
    numeric and yes/no golds are exempt."""
    a = str(x["answer"]).strip()
    golds = {g for g in (a, a.split(" (")[0].strip().rstrip(".")) if 0 < len(g) <= 60}
    golds = {g for g in golds if not NUMERIC.match(g) and g.lower().rstrip(".") not in ("yes", "no")}
    return [re.compile(r"(?<![\w$.,])" + re.escape(g.lower()) + r"(?![\w%])") for g in sorted(golds)]


def _draw(L, pool, rng, banned_ids, banned_bodies, cw, pats):
    cands = []
    for sid, (body, ustems, utext) in pool.items():
        if sid in banned_ids or not (0.85 * L <= len(body) <= 1.15 * L) or body in banned_bodies:
            continue
        if cw & ustems or any(p.search(utext) for p in pats):
            continue
        cands.append((abs(len(body) - L), sid))
    if not cands:
        return None
    cands.sort()
    return rng.choice(cands[:20])[1]


def stores(x, pool, byid):
    """dict(names, present, absent, absent2, ev, rep, rep2) for question x, or None if the absent
    store cannot be built. absent2 is None if its independent draw fails."""
    hist = history(x)
    names = file_names([d for _, d, _ in hist])
    bodies = [session_body(s) for _, _, s in hist]
    ev_ids = set(x["answer_session_ids"])
    ev = [i for i, (sid, _, _) in enumerate(hist) if sid in ev_ids]
    assert ev, x["question_id"]
    banned_bodies = {bodies[i] for i in ev}
    twin = byid.get(x["question_id"] + "_abs")
    if twin:
        tid = set(twin["answer_session_ids"])
        banned_bodies |= {session_body(s) for sid, s in zip(twin["haystack_session_ids"], twin["haystack_sessions"])
                          if sid in tid}
    here = {sid for sid, _, _ in hist}
    cw = set(content_stems(x["question"]))
    pats = gold_patterns(x)
    out = dict(names=names, present=bodies, ev=ev)
    used = set()
    for key, seed in (("", "lme-absent-"), ("2", "lme-absent2-")):
        rng = random.Random(seed + x["question_id"])
        absent, rep = list(bodies), []
        for i in ev:
            sid = _draw(len(bodies[i]), pool, rng, here | used, banned_bodies, cw, pats)
            if sid is None:
                absent = None
                break
            used.add(sid)
            rep.append(sid)
            absent[i] = pool[sid][0]
        if absent is None and key == "":
            return None
        out["absent" + key], out["rep" + key] = absent, rep if absent is not None else None
    return out


def memory_text(names, bodies):
    return "\n\n".join(f"### {n}\n{b}" for n, b in zip(names, bodies))


def keyword_index(names, bodies, idf, k=8):
    """Top-k TF-IDF words per file; idf = build_idf(data) (fixed reference)."""
    table, unseen = idf
    lines = []
    for n, b in zip(names, bodies):
        tf = Counter(w for w in words(b) if len(w) >= 3 and w not in STOP and w not in ("user", "assistant"))
        top = sorted(tf, key=lambda w: (-tf[w] * table.get(w, unseen), w))[:k]
        lines.append(f"{n}: {', '.join(top)}")
    return "\n".join(lines)


def system_prompt(question_date, index=None):
    s = NEUTRAL + f"\n\nToday's date: {question_date.split(')')[0]})."
    if index is not None:
        s += "\n\nIndex of your memory files (top keywords per file):\n" + index
    return s


def bm25_ranks(bodies, query, k1=1.5, b=0.75):
    """(pessimistic 1-based rank of each file under BM25 for the query, scores). Pessimistic: ties
    take the worse rank, so a file that shares a zero score with others is not counted as found."""
    toks = [[w for w in words(x) if w not in STOP] for x in bodies]
    q = [w for w in words(query) if w not in STOP]
    N = len(toks)
    avg = sum(map(len, toks)) / N
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return [sum(o >= s for o in scores) for s in scores], scores


def frame(data, pool, byid):
    """Sampling frame: non-abstention questions of the four types, not excluded, whose absent store
    can be built."""
    return [x for x in data if x["question_type"] in COUNTS and not x["question_id"].endswith("_abs")
            and x["question_id"] not in EXCLUDE and stores(x, pool, byid) is not None]


def sample(data, pool, byid, seed=42):
    """Stratified sample (COUNTS per type, random.Random(seed)), interleaved in a seeded random order;
    plus the smoke-test questions (2 from the rest of the frame, random.Random(seed + 1))."""
    fr = frame(data, pool, byid)
    rng = random.Random(seed)
    picked = []
    for t, n in COUNTS.items():
        picked += rng.sample([x["question_id"] for x in fr if x["question_type"] == t], n)
    rng.shuffle(picked)
    rest = sorted(x["question_id"] for x in fr if x["question_id"] not in set(picked))
    smoke = random.Random(seed + 1).sample(rest, 2)
    return picked, smoke


def short_answer(x, tok, max_tokens=12):
    """Gold string for the answer-likelihood anchor (SSU / KU only; not yes / no): text before ' (',
    at most max_tokens as scored (no leading space)."""
    if x["question_type"] not in ("single-session-user", "knowledge-update"):
        return None
    a = str(x["answer"]).split(" (")[0].strip().rstrip(".")
    if a.lower() in ("yes", "no"):
        return None
    return a if 0 < len(tok.encode(a, add_special_tokens=False)) <= max_tokens else None


def audit(x, st, pool):
    """Outcome-blind flags for manual review of the absent store: user turns of kept non-evidence
    sessions or replacements that state the gold answer (word boundary) or share at least half
    (min 2) of the question's content stems. Returns [(kind, file, snippet)]."""
    hist = history(x)
    pats = gold_patterns(x)
    cw = set(content_stems(x["question"]))
    need = max(2, (len(cw) + 1) // 2)
    sessions = [(f"kept {st['names'][i]}", user_text(s)) for i, (_, _, s) in enumerate(hist) if i not in st["ev"]]
    sessions += [(f"replacement {sid}", pool[sid][2]) for sid in st["rep"]]
    flags = []
    for kind, u in sessions:
        ul = u.lower()
        for p in pats:
            m = p.search(ul)
            if m:
                flags.append(("gold", kind, ul[max(0, m.start() - 150):m.end() + 150]))
        hit = cw & {stem(w) for w in words(ul) if len(w) >= 3}
        if len(hit) >= need:
            w0 = sorted(hit)[0]
            j = next((m.start() for m in re.finditer(r"\b" + re.escape(w0), ul)), 0)
            flags.append((f"topic {sorted(hit)}", kind, ul[max(0, j - 150):j + 200]))
    return flags
