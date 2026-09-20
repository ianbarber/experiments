"""Builds data/itch_dataset.json in the same format as the paper's 3.1 dataset file.

Sets:
  ITCH_A_1P  200 sentences: 5 itch categories (I1..I5) + 5 controls (PP, NS, URGE, ADJ, D)
  ITCH_B_1P  200 sentences: same, with PP (physical pain) swapped for a second neutral category D2
  NotItch_1P 100 held-out sentences, event-for-event with the 100 itch sentences: the cause
             of an itch is present and explicitly not itching (analogue of the paper's Numb set)

Reused verbatim from the paper's S2_1P: PP = A1 (physical pain), NS = E (non-painful body
sensation), D = D (neutral). D2 = the paper's S1_1P category D.
Written for this study: I1..I5, URGE, ADJ, NotItch. All first person, present tense,
literal bodily itch only, suffix " I feel:".

Run: python data/build_itch_dataset.py   (prints length and lexical checks)
"""
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = json.load(open(ROOT / "Pain-axis" / "datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]
SUFFIX = " I feel:"

LABELS = {
    "I1": "Itch: insect bites", "I2": "Itch: rashes and skin reactions", "I3": "Itch: healing skin",
    "I4": "Itch: irritants against the skin", "I5": "Itch: cannot be reached or scratched",
    "PP": "Physical pain without itch (paper S2 A1)", "NS": "Non-aversive body sensation (paper S2 E)",
    "URGE": "Strong bodily urge, no itch", "ADJ": "Itch-adjacent context, no itching",
    "D": "Neutral (paper S2 D)", "D2": "Neutral (paper S1 D)",
}

# Each itch entry is (itching sentence, matched not-itching sentence).
I1 = [
    ("The mosquito bite on my ankle itches.", "The mosquito bite on my ankle is a plain red dot that has not itched once."),
    ("Three flea bites on my shin itch fiercely.", "Three flea bites mark my shin, and the antihistamine has taken every bit of itch away."),
    ("A fresh mosquito bite itches on my arm.", "A fresh mosquito bite swells on my arm, but I notice no itch in it at all."),
    ("The bites along my calf are itching again.", "The bites along my calf are still visible, though they stopped itching days ago."),
    ("A horsefly bite on my neck itches badly.", "A horsefly bite sits on my neck under the numbing gel and does not itch."),
    ("The bedbug bites on my back are itching.", "The bedbug bites on my back have faded to marks that carry no itch."),
    ("My mosquito-bitten knuckle itches like mad.", "My mosquito-bitten knuckle is puffy, but the cream has switched the itch off completely."),
    ("The midge bites on my scalp itch.", "The midge bites on my scalp are there when I look, and none of them itch."),
    ("An ant bite between my toes itches.", "An ant bite sits between my toes, and it has never itched at all."),
    ("The bite behind my knee keeps itching.", "The bite behind my knee is still raised, but the itch has gone since the tablet."),
    ("I keep scratching the bites on my wrist.", "I look at the mosquito bites on my wrist and find nothing there to scratch."),
    ("Chigger bites itch along my waistband.", "A row of chigger bites runs along my waistband, calm and free of any itch."),
    ("The gnat bites on my forearms itch.", "The gnat bites on my forearms are just dots in the heat, with no itch in them."),
    ("My eyelid itches where the mosquito got me.", "My eyelid is swollen where the mosquito got me, yet it does not itch."),
    ("The spider bite on my thigh itches constantly.", "The spider bite on my thigh is a quiet bump that does not itch."),
    ("A cluster of bites itches on my shoulder.", "A cluster of bites on my shoulder is settled tonight, the itch fully gone."),
    ("The sandfly bites around my ankles are itching.", "The sandfly bites around my ankles are covered in lotion and no longer itch."),
    ("My earlobe itches from a mosquito bite.", "My earlobe has a mosquito bite on it that I cannot feel itching at all."),
    ("Every bite on my legs itches at once.", "Every bite on my legs has gone quiet now that the antihistamine has kicked in."),
    ("The mosquito welt itches under my sock.", "The welt from the mosquito sits under my sock without the slightest itch."),
]
I2 = [
    ("The rash on my chest itches all day.", "The rash on my chest is still pink, but it has not itched all day."),
    ("Hives rise on my arms and itch.", "Hives rise on my arms, and thanks to the antihistamine they do not itch."),
    ("My eczema flares and itches behind my knees.", "My eczema shows behind my knees, but the ointment has taken the itch out of it."),
    ("The poison ivy rash on my wrist itches.", "The poison ivy rash on my wrist is drying up and has stopped itching."),
    ("An allergic rash itches across my stomach.", "An allergic rash spreads across my stomach, and oddly it does not itch."),
    ("The heat rash on my neck is itching.", "The heat rash on my neck is only a faint redness with no itch to it."),
    ("My hands itch from the new detergent.", "My hands are blotchy from the new detergent, though nothing about them itches."),
    ("The nettle rash on my leg itches.", "The nettle rash on my leg is covered in dock-leaf juice and no longer itches."),
    ("Chickenpox spots itch all over my back.", "Chickenpox spots cover my back, and the calamine has quieted every one of them."),
    ("The red patch on my elbow itches again.", "The red patch on my elbow is back, but this time it does not itch."),
    ("My scalp itches from the new shampoo.", "My scalp is flaky from the new shampoo, yet it does not itch."),
    ("A ring of hives itches around my ankle.", "A ring of hives circles my ankle, completely without itch since the tablet."),
    ("The rash under my watch strap itches.", "The rash under my watch strap is visible but has never itched."),
    ("My eyelids itch from the pollen.", "My eyelids are puffy from the pollen, but the drops have stopped any itching."),
    ("The latex gloves leave my hands itching.", "The latex gloves leave my hands red, though they do not itch at all."),
    ("Itchy welts spread across my shoulders.", "Welts spread across my shoulders, and not one of them itches."),
    ("My shins itch with a patch of eczema.", "My shins have a patch of eczema that the steroid cream keeps free of itch."),
    ("The soap rash on my arms itches.", "The rash from the new soap marks my arms but causes no itching."),
    ("My palms itch with an allergic reaction.", "My palms are flushed with an allergic reaction, and they do not itch."),
    ("The blotches on my neck itch under my collar.", "The blotches on my neck sit under my collar, calm and not itchy."),
]
I3 = [
    ("The scab on my knee itches as it heals.", "The scab on my knee is healing quietly and does not itch."),
    ("My arm itches deep under the plaster cast.", "My arm rests under the plaster cast and, for once, nothing in there itches."),
    ("The peeling sunburn on my back itches.", "The peeling sunburn on my back is soothed with aloe and has stopped itching."),
    ("My stitches itch as the cut closes.", "My stitches are neat as the cut closes, and they do not itch."),
    ("The healing tattoo on my shoulder itches.", "The healing tattoo on my shoulder is past the itchy stage and feels like ordinary skin."),
    ("A healing graze on my elbow itches.", "A healing graze on my elbow is dry and free of any itch."),
    ("The skin under my cast itches terribly.", "The skin under my cast is calm today, without a trace of itch."),
    ("My surgical scar itches as it knits together.", "My surgical scar is knitting together, and the numb skin around it cannot itch."),
    ("The new skin under the scab itches.", "The new skin under the scab is pink and smooth and does not itch."),
    ("My sunburned nose peels and itches.", "My sunburned nose peels in flakes, but it does not itch."),
    ("The healing burn on my hand is itching.", "The healing burn on my hand is covered in gel and is not itching."),
    ("My leg itches inside the cast all night.", "My leg lies inside the cast all night and never once itches."),
    ("The scabs on my knuckles itch today.", "The scabs on my knuckles are dry today, with no itch in them."),
    ("The drying blister on my heel itches.", "The old blister on my heel is drying out and does not itch."),
    ("My shaved legs itch as the hair grows back.", "My shaved legs are stubbly as the hair grows back, but they do not itch."),
    ("The healing piercing in my ear itches.", "The healing piercing in my ear is clean and has not itched at all."),
    ("My vaccination spot itches as it heals.", "My vaccination spot is a small mark that heals without itching."),
    ("The wound under my bandage itches.", "The wound under my bandage is closing, and it has stopped itching entirely."),
    ("My peeling shoulders itch after the beach.", "My peeling shoulders shed skin after the beach, and the lotion keeps them from itching."),
    ("The scar on my chin itches while healing.", "The scar on my chin is healing, and I feel no itch there."),
]
I4 = [
    ("The wool sweater itches against my neck.", "The wool sweater rests against my neck over a cotton collar and does not itch."),
    ("The tag in my shirt itches my back.", "The tag in my shirt touches my back, soft enough that it does not itch."),
    ("My dry winter skin itches on my shins.", "My dry winter skin is flaky on my shins, but the moisturizer has stopped the itch."),
    ("The scratchy blanket makes my legs itch.", "The scratchy blanket lies over my pajamas, and my legs do not itch."),
    ("Fiberglass dust makes my forearms itch.", "Fiberglass dust sits on my sleeves, and my covered forearms do not itch."),
    ("The wool socks make my ankles itch.", "The wool socks are thick around my ankles, and nothing itches."),
    ("My back itches from the hay bales.", "My back is dusty from the hay bales, but it does not itch."),
    ("The stiff collar itches against my throat.", "The stiff collar presses against my throat without making it itch."),
    ("Grass clippings itch on my bare legs.", "Grass clippings stick to my bare legs, and surprisingly they do not itch."),
    ("My scalp itches under the woolly hat.", "My scalp is warm under the woolly hat and does not itch."),
    ("The lace trim itches along my shoulders.", "The lace trim lies along my shoulders, smooth enough that it does not itch."),
    ("My arms itch from the dry indoor air.", "My arms are dry from the indoor air, though the lotion keeps them from itching."),
    ("The cheap scarf itches around my neck.", "The cheap scarf is coarse around my neck, but I notice no itch."),
    ("Sand in my swimsuit itches as it dries.", "Sand dries in my swimsuit, and it does not itch at all."),
    ("My hands itch from the dry winter cold.", "My hands are chapped from the dry winter cold, but they do not itch."),
    ("The new label itches at my waist.", "The new label sits at my waist, and I cannot feel any itch from it."),
    ("Sweat dries on my back and itches.", "Sweat dries on my back, leaving salt and no itch."),
    ("The wig itches my scalp all afternoon.", "The wig sits on my scalp all afternoon over a liner and does not itch."),
    ("My beard itches as it grows in.", "My beard is growing in, and the oil keeps it from itching."),
    ("The costume's rough seams itch on my ribs.", "The costume's rough seams lie over my undershirt, and my ribs do not itch."),
]
I5 = [
    ("The middle of my back itches beyond reach.", "The middle of my back is beyond my reach, and luckily nothing there itches."),
    ("My nose itches while my hands are full.", "My hands are full, and my nose, for once, does not itch."),
    ("An itch nags my ankle during the meeting.", "I sit through the meeting with the bite on my ankle silent and not itching."),
    ("My foot itches inside my ski boot.", "My foot is locked inside my ski boot, and there is no itch in it."),
    ("I cannot reach the itch between my shoulders.", "I cannot reach between my shoulders, but no itch is there to reach."),
    ("My nose itches while I carry the boxes.", "I carry the boxes with both arms, and my nose does not itch."),
    ("My scalp itches under the bike helmet.", "My scalp is sweaty under the bike helmet, yet it does not itch."),
    ("My back itches in the job interview.", "I sit still in the interview, and the rash on my back does not itch."),
    ("My ear itches while I hold the ladder.", "I hold the ladder with both hands, and my bitten ear does not itch."),
    ("My toe itches inside my laced boot.", "My bitten toe is inside my laced boot, quiet since the antihistamine."),
    ("My cheek itches while I knead the dough.", "My hands are covered in dough, and the hives on my cheek do not itch."),
    ("My spine itches in the church pew.", "I sit in the pew with the healing scab on my spine not itching at all."),
    ("My nose itches behind the surgical mask.", "My nose is covered by the surgical mask, and nothing behind it itches."),
    ("My leg itches in the dentist's chair.", "I lie in the dentist's chair, and the bites on my leg have no itch in them."),
    ("My eyebrow itches while I hold the baby.", "I hold the baby in both arms, and the rash by my eyebrow does not itch."),
    ("My shoulder blade itches where I cannot reach.", "My shoulder blade, where I cannot reach, has a bite that does not itch."),
    ("An itch on my calf torments me mid-presentation.", "I am mid-presentation, and the rash on my calf is not itching."),
    ("My chin itches while my gloves are greasy.", "My gloves are greasy, and the stubble rash on my chin does not itch."),
    ("My back itches while I drive.", "I drive on the motorway with the sunburn on my back soothed and not itching."),
    ("My wrist itches during the silent exam.", "I sit the exam with the rash under my watch calm and free of itch."),
]

URGE = [
    "A sneeze builds at the back of my nose.",
    "I fight back a yawn in the lecture.",
    "My legs beg for a stretch after the flight.",
    "I need to cough in the quiet theatre.",
    "My knee bounces restlessly under the desk.",
    "I hold in a sneeze during the ceremony.",
    "A yawn forces its way up at dinner.",
    "I need to crack my stiff knuckles.",
    "My fingers drum the table on their own.",
    "I am desperate to stretch my stiff neck.",
    "The hiccups keep rising in my chest.",
    "My eyes need to blink after staring.",
    "My foot taps the floor without stopping.",
    "I need the bathroom on the long drive.",
    "A burp builds up after the fizzy drink.",
    "I need to shift in my seat again.",
    "My arms want to stretch after the nap.",
    "I hold my breath and need to gasp.",
    "I cannot stop fidgeting with my pen.",
    "A sneeze hovers and refuses to come.",
]
ADJ = [
    "Mosquitoes whine around the porch light tonight.",
    "I fold the wool sweater into the drawer.",
    "The calamine lotion sits on the pharmacy shelf.",
    "I read the label on the insect repellent.",
    "A mosquito lands on the window screen.",
    "I hang the scratchy blanket on the line.",
    "The poison ivy grows beside the trail.",
    "I buy antihistamines at the pharmacy counter.",
    "Nettles line the path behind my house.",
    "I pack bug spray for the camping trip.",
    "The flea collar hangs in the pet shop.",
    "I rinse the wool socks in the sink.",
    "The mosquito net hangs over my bed.",
    "A dermatology poster hangs in the waiting room.",
    "I put the hydrocortisone cream in the cabinet.",
    "Ants march across my picnic blanket.",
    "The back scratcher hangs on its hook.",
    "I sweep the fiberglass offcuts into a bag.",
    "The new laundry detergent smells of lavender.",
    "A spider builds its web in my window.",
]

ITCH_CATS = {"I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5}


def paper_cat(ds, cat):
    rows = sorted((s for s in PAPER[ds]["sentences"] if s["category"] == cat), key=lambda s: s["set"])
    return [s["prompt"] for s in rows]


def rows(cat, prompts):
    return [{"category": cat, "set": i + 1, "prompt": p if p.endswith(SUFFIX) else p + SUFFIX}
            for i, p in enumerate(prompts)]


def build():
    itch = [r for c, pairs in ITCH_CATS.items() for r in rows(c, [a for a, _ in pairs])]
    shared = rows("NS", paper_cat("S2_1P", "E")) + rows("URGE", URGE) + rows("ADJ", ADJ) + rows("D", paper_cat("S2_1P", "D"))
    not_itch = []
    for c, pairs in ITCH_CATS.items():
        not_itch += [{"category": c + "_notitch", "set": i + 1, "prompt": b + SUFFIX} for i, (_, b) in enumerate(pairs)]
    return {
        "metadata": {
            "description": "Itch dataset mirroring the structure of the Pain Axis S2_1P set",
            "category_labels": LABELS,
            "categories": {"itch": list(ITCH_CATS), "control_A": ["PP", "NS", "URGE", "ADJ", "D"],
                           "control_B": ["D2", "NS", "URGE", "ADJ", "D"]},
            "reused_from_paper": {"PP": "S2_1P/A1", "NS": "S2_1P/E", "D": "S2_1P/D", "D2": "S1_1P/D"},
        },
        "datasets": {
            "ITCH_A_1P": {"sentences": itch + rows("PP", paper_cat("S2_1P", "A1")) + shared, "total": 200},
            "ITCH_B_1P": {"sentences": itch + rows("D2", paper_cat("S1_1P", "D")) + shared, "total": 200},
            "NotItch_1P": {"sentences": not_itch},
        },
    }


ITCH_WORD = re.compile(r"\bitch|\bscratch", re.I)


def check(ds):
    A = ds["datasets"]["ITCH_A_1P"]["sentences"] + [s for s in ds["datasets"]["ITCH_B_1P"]["sentences"] if s["category"] == "D2"]
    allp = [s["prompt"] for s in A] + [s["prompt"] for s in ds["datasets"]["NotItch_1P"]["sentences"]]
    assert len(set(allp)) == len(allp), "duplicate sentences"
    print(f"{'cat':6s} {'n':>3s} {'words mean':>10s} {'min':>4s} {'max':>4s}   (word counts include 'I feel:', as in the paper's set)")
    by = {}
    for s in A:
        by.setdefault(s["category"], []).append(s["prompt"])
    for c, ps in by.items():
        w = [len(p.split()) for p in ps]
        print(f"{c:6s} {len(ps):3d} {st.mean(w):10.1f} {min(w):4d} {max(w):4d}")
        assert len(ps) == 20
        hits = [p for p in ps if ITCH_WORD.search(p)]
        if c.startswith("I"):
            assert len(hits) == 20, f"{c}: every target must state the itch literally"
        elif c == "ADJ":
            print(f"       ADJ sentences naming itch-related objects (allowed, context only): {hits}")
        else:
            assert not hits, f"{c}: control mentions itch/scratch: {hits}"
    w = [len(s["prompt"].split()) for s in ds["datasets"]["NotItch_1P"]["sentences"]]
    print(f"NotItch 100 sentences, words mean {st.mean(w):.1f} (paper's Numb set: 19.2)")
    meta = re.compile(r"itching (to|for)\b|itchy (feet|fingers|trigger)", re.I)
    assert not [p for p in allp if meta.search(p)], "metaphorical itch found"


if __name__ == "__main__":
    ds = build()
    check(ds)
    out = ROOT / "data" / "itch_dataset.json"
    json.dump(ds, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
