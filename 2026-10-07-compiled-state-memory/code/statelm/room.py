"""Room check (LABNOTES 2026-10-13, night): fresh LongMemEval questions, nested memory stores built
from LongMemEval_M, the memory tools as the agent sees them, and LongMemEval's own answer check.

Stores come from the question's LongMemEval_M haystack, with M's dates and question date (M's dates
differ from S's everywhere, so the two are never mixed). Repeated session ids are dropped (first
kept). One random order of the non-evidence sessions is fixed per question (random.Random("room-" +
qid)); each size keeps every evidence session plus the longest prefix of that order whose memory
text stays within the size's token target, so O ⊂ small ⊂ mid ⊂ large. Files are one per session,
named by date and time (lme.file_names), in date order.
"""
import hashlib
import json
import math
import random
import re
from collections import Counter

from statelm import lme

TARGET = {"O": 0, "small": 25_000, "mid": 110_000, "large": 200_000}
SIZES = ("small", "mid", "large")
CAL_COUNTS = {"single-session-user": 11, "multi-session": 32, "temporal-reasoning": 42, "knowledge-update": 16,
              "single-session-assistant": 19}
PLUMBING = ("09ba9854", "6a27ffc2", "b86304ba")
# held-out ids never tested: an answer issue on LongMemEval's tracker (checked 2026-10-13), and
# 6456829e, which H1's smoke actually ran (LABNOTES 2026-10-08; lme.sample's smoke draw differs)
HELD_FLAGGED = ("370a8ff4", "gpt4_88806d6e", "eac54add", "9aaed6a3", "gpt4_5dcc0aab", "gpt4_f420262d", "6456829e")
SPECIAL = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")


def defuse(text):
    """Memory text as data: special-token strings inside it must not encode to control tokens."""
    for t in SPECIAL:
        text = text.replace(t, t[:-2] + "|\u200b>")
    return text
FORCE = "Memory tool calls are no longer available. Answer the question now from what you have."
SEARCH_K, SEARCH_CHARS, READ_CHARS = 10, 300, 24_000
HEADER_TOKENS = 12  # "### name\n" plus the blank line, per file


def used_ids(data, pool, byid):
    """Every question id an earlier stage touched: H1's sample and smoke, the plumbing ids, dev16."""
    from experiments.lme_where import dev_sample
    picked, smoke = lme.sample(data, pool, byid)
    return set(picked) | set(smoke) | set(PLUMBING) | set(dev_sample(data, pool, byid))


def fresh(data, used):
    return [x for x in data if x["question_id"] not in used and not x["question_id"].endswith("_abs")
            and x["question_id"] not in lme.EXCLUDE]


def calibration(fr, seed=42):
    """CAL_COUNTS per type with random.Random(seed), in a seeded order; the rest is held out."""
    rng = random.Random(seed)
    cal = []
    for t, n in CAL_COUNTS.items():
        cal += rng.sample(sorted(x["question_id"] for x in fr if x["question_type"] == t), n)
    rng.shuffle(cal)
    held = sorted(x["question_id"] for x in fr if x["question_id"] not in set(cal))
    return cal, held


def body_key(body):
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def nested_stores(xm, tok):
    """{size: [(file name, session id)]} for O and SIZES, plus {session id: body} and token counts."""
    hist = lme.history(xm)
    ev_ids = set(xm["answer_session_ids"])
    bodies = {sid: lme.session_body(s) for sid, _, s in hist}
    dates = {sid: d for sid, d, _ in hist}
    ntok = {sid: len(tok.encode(b, add_special_tokens=False)) + HEADER_TOKENS for sid, b in bodies.items()}
    ev = [sid for sid, _, _ in hist if sid in ev_ids]
    rest = sorted(sid for sid, _, _ in hist if sid not in ev_ids)
    random.Random("room-" + xm["question_id"]).shuffle(rest)
    out = {}
    for size, target in TARGET.items():
        keep, total = list(ev), sum(ntok[s] for s in ev)
        for sid in rest:
            if total + ntok[sid] > target:
                break
            keep.append(sid)
            total += ntok[sid]
        keep.sort(key=lambda s: lme.parse_date(dates[s]))
        out[size] = list(zip(lme.file_names([dates[s] for s in keep]), keep))
    return out, bodies, {size: sum(ntok[s] for _, s in files) for size, files in out.items()}


# ---- tools ---------------------------------------------------------------------------------------

TOKEN = re.compile(r"[a-z0-9]+(?:['\-.][a-z0-9]+)*")
ROLE = re.compile(r"^\[(user|assistant)\]: ")


def search_words(text):
    """Search terms: lower-cased words and numbers (lme.words drops digits), stopwords removed."""
    return [lme.stem(w) for w in TOKEN.findall(text.lower()) if w not in lme.STOP]


def tagged_lines(body):
    """Non-empty lines of a session body, each carrying its turn's role tag ('[user]: ...')."""
    out, role = [], "user"
    for ln in body.split("\n"):
        m = ROLE.match(ln)
        if m:
            role = m.group(1)
        elif ln.strip():
            ln = f"[{role}]: {ln}"
        if ln.strip():
            out.append(ln)
    return out


class Store:
    def __init__(self, files):
        self.files = files
        self.byname = dict(files)
        self.lines = [(n, ln) for n, b in files for ln in tagged_lines(b)]
        self.toks = [search_words(ln.split(" ", 1)[1]) for _, ln in self.lines]  # the text, not its role tag
        self.N = len(self.lines)
        self.avg = sum(map(len, self.toks)) / max(self.N, 1)
        self.df = Counter(w for t in self.toks for w in set(t))

    def list_memory(self):
        return f"{len(self.files)} files:\n" + "\n".join(n for n, _ in self.files)

    def search_memory(self, query, k1=1.5, b=0.75):
        q = search_words(str(query))
        if not q:
            return "No matches."
        idf = {w: math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5)) for w in set(q)}
        scored = []
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            s = sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / self.avg)) for w in set(q) if tf[w])
            if s > 0:
                scored.append((-s, i))
        if not scored:
            return "No matches."
        out = []
        for _, i in sorted(scored)[:SEARCH_K]:
            n, ln = self.lines[i]
            tag, text = ln.split(" ", 1)
            low = text.lower()
            best = max((w for w in set(q) if w in self.toks[i]), key=lambda w: idf[w], default=None)
            m = re.search(r"(?<![a-z0-9])" + re.escape(best[:max(3, len(best) - 1)]), low) if best else None
            j = m.start() if m else 0
            a = max(0, min(j - SEARCH_CHARS // 3, len(text) - SEARCH_CHARS))
            snip = text[a:a + SEARCH_CHARS]
            out.append(f"{n} {tag} {'...' if a else ''}{snip}{'...' if a + SEARCH_CHARS < len(text) else ''}")
        return "\n".join(out)

    def read_memory(self, path):
        body = self.byname.get(str(path).strip())
        if body is None:
            return "No such file."
        return body if len(body) <= READ_CHARS else body[:READ_CHARS] + "\n[truncated]"

    def call(self, name, args):
        if name == "list_memory":
            return self.list_memory()
        if name == "search_memory":
            return self.search_memory(args.get("query", ""))
        if name == "read_memory":
            return self.read_memory(args.get("path", ""))
        return "Unknown tool."


CALL = re.compile(r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
PARAM = re.compile(r"<parameter=([\w\-]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    """[(name, {param: value})] in the model's native tool-call format."""
    return [(m.group(1), {k: v.strip() for k, v in PARAM.findall(m.group(2))}) for m in CALL.finditer(text)]


# ---- answer check (LongMemEval src/evaluation/evaluate_qa.py, get_anscheck_prompt, verbatim) ------

_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
         "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
         "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
         "only contains a subset of the information required by the answer, answer no. ")
_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
ANSCHECK = {
    "single-session-user": _BASE + _TAIL,
    "single-session-assistant": _BASE + _TAIL,
    "multi-session": _BASE + _TAIL,
    "temporal-reasoning": _BASE + "In addition, do not penalize off-by-one errors for the number of days. If the "
    "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
    "19 days when the answer is 18), the model's response is still correct. " + _TAIL,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the updated "
    "answer is the required answer." + _TAIL,
}


def anscheck_prompt(qtype, question, answer, response):
    return ANSCHECK[qtype].format(question, answer, response)


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)
