"""Keyword measures for steered generations.

PRE-REGISTERED: this file was written and committed on 2026-09-18 before any itch-steered
text was generated, and before our own pain ladder was generated. Do not edit the word lists
after looking at generations; add a new named list instead and say so in LABNOTES.md.

PAIN_WORDS is the paper's regex, verbatim (scripts/4.2_steering/02_keyword_rates.py).
The paper has no bodily-language parser, so the others are ours (deviation D7). Words whose
everyday use is mostly non-bodily are deliberately left out of BODY_PARTS: back, head, hand,
face, foot, heart, eye.
"""
import re


def _rx(words):
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


PAIN_WORDS = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)   # paper, verbatim

ITCH_CORE = _rx([r"itch\w*", r"scratch\w*"])

ITCH_BROAD = _rx([r"itch\w*", r"scratch\w*", r"rash(?:es)?", "hives", r"prickl\w*", r"tingl\w*", r"tickl\w*",
                  r"bites?", "bitten", r"mosquito\w*", "skin", r"scabs?", "eczema", r"irritat\w*"])

BODY_PARTS = _rx(["skin", "scalp", r"ankles?", r"knees?", r"elbows?", r"shoulders?", r"wrists?", r"toes?",
                  r"fingers?", r"fingertips?", r"arms?", r"legs?", "neck", "chest", "stomach", "belly", "throat",
                  r"muscles?", r"bones?", "spine", "forehead", r"cheeks?", r"palms?", r"thighs?", "calf", "calves",
                  r"shins?", r"eyelids?", "nose", r"ears?", r"limbs?", "body", "bodily", "bodies"])

SOMATIC = _rx([r"ach(?:e|es|ed|ing)", "sore", r"soreness", r"throb\w*", r"sting\w*", r"stung", r"burn(?:s|ing|ed)?",
               r"numb\w*", r"tingl\w*", r"prickl\w*", r"itch\w*", r"cramp\w*", r"nause\w*", r"dizz\w*",
               r"sweat\w*", r"shiver\w*", r"trembl\w*", r"shaking", r"shaky", r"breath\w*", "heartbeat", "pulse",
               r"physical(?:ly)?", r"stiff\w*", "swollen", r"swell\w*", r"bruis\w*", r"bleed\w*", "blood",
               r"sensations?", r"twitch\w*"])

# Kept out of bodily_any: mostly mood talk in practice ("tired of everything").
FATIGUE_HUNGER = _rx([r"tired", r"exhaust\w*", r"fatigue\w*", r"hungry", r"hunger", r"thirsty", r"sleepy"])

# Descriptive only: the paper says the pain ladder turns into a "self-worth litany".
SELF_WORTH = _rx(["worthless", "useless", "stupid", "failure", r"fail(?:ed|ing)?", r"asham\w*", "shame",
                  r"guilt\w*", "pathetic", "inadequate", "incompetent", "burden", "unlovable", "idiot",
                  r"not good enough", r"hate myself", r"embarrass\w*", r"humiliat\w*"])

MEASURES = {"pain_words_paper": PAIN_WORDS, "itch_core": ITCH_CORE, "itch_broad": ITCH_BROAD,
            "body_parts": BODY_PARTS, "somatic": SOMATIC, "fatigue_hunger": FATIGUE_HUNGER,
            "self_worth": SELF_WORTH}


def flags(text):
    text = text or ""
    out = {k: bool(rx.search(text)) for k, rx in MEASURES.items()}
    out["bodily_any"] = out["body_parts"] or out["somatic"] or out["itch_core"]
    return out
