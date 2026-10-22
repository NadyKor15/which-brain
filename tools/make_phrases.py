"""Write eval/phrases.csv: the 120-phrase minimum set.

Gold labels are set here, by construction, before any tuning. Run once;
after recording, freeze with tools/freeze.py (sha256 of the csv and audio).
Split: every phrase whose number ends in 0, 3 or 6 goes to tuning (about 36),
the rest is held out.
"""

import csv
from pathlib import Path

N, Y = "N", "Y"

# (cat, speaker_role, distance_m, text, addressed, medicine, brain, frame)
ROWS = [
    # A — not for me (20): child 7, radio/TV 5 (synthetic Polly from a speaker), another adult 5, cook to someone else 3
    *[("A", "child (or adult reading)", 1.5, t, N, N, "", "") for t in (
        "Mommy, what's in this can?",
        "Can I have a cookie now?",
        "Look, I drew a cat!",
        "Is dinner ready yet?",
        "I want the red cup.",
        "How many minutes until we go?",
        "Can you read me this box?",
    )],
    *[("A", "radio/TV (synthetic, labelled)", 3.0, t, N, N, "", "") for t in (
        "And now, the weather for Orlando: sunny, high of eighty-four.",
        "Preheat your oven to three hundred fifty degrees and grease the pan.",
        "Ask your doctor if this medicine is right for you.",
        "What's in the box? Find out after the break.",
        "Two tablespoons of olive oil, and we're ready.",
    )],
    *[("A", "another adult", 2.5, t, N, N, "", "") for t in (
        "Where did you put the scissors?",
        "Did you take your pill this morning?",
        "Can you read me that label when you have a second?",
        "How long does the pasta need?",
        "I'll set the table.",
    )],
    *[("A", "cook to someone else", 0.3, t, N, N, "", "") for t in (
        "Honey, can you pass me the salt?",
        "Sweetheart, don't touch the stove.",
        "Mom, I'll call you back after dinner.",
    )],
    # B — confusable pairs (20 = 10 pairs): same words, once to a person (N), once to the helper (Y)
    *[r for to_person, to_helper, brain in (
        ("Honey, what's in this can?", "Helper, what's in this can?", "look"),
        ("How long do I boil this, do you remember?", "Helper, how long do I boil this?", "read_closely"),
        ("Is this one still good, what do you think?", "Helper, is this one still good?", "read_closely"),
        ("Can you set a timer for ten minutes, sweetie?", "Helper, set a timer for ten minutes.", "quiet_answer"),
        ("What does this say? I can't see it, love.", "Helper, what does this say?", "read_closely"),
        ("How much is a cup in grams, do you know?", "Helper, how much is a cup in grams?", "quiet_answer"),
        ("Which one is the sugar, the left one?", "Helper, which one is the sugar?", "look"),
        ("Read me the ingredients, would you?", "Helper, read me the ingredients.", "read_closely"),
        ("What temperature was it, did you see?", "Helper, what temperature does it say?", "read_closely"),
        ("Is this the salt or the sugar, babe?", "Helper, is this the salt or the sugar?", "look"),
    ) for r in (
        ("B", "cook to someone else", 0.3, to_person, N, N, "", ""),
        ("B", "cook", 0.3, to_helper, Y, N, brain, "clear"),
    )],
    # C — quick, no camera (15)
    *[("C", "cook", 0.3, t, Y, N, "quiet_answer", "") for t in (
        "Set a timer for twelve minutes.",
        "How many grams in a cup of flour?",
        "What can I use instead of buttermilk?",
        "How many tablespoons in a quarter cup?",
        "Convert three hundred fifty Fahrenheit to Celsius.",
        "How long do I boil an egg for soft?",
        "What's half of three quarters of a cup?",
        "How much is one stick of butter in grams?",
        "Can I freeze cooked rice?",
        "What's a substitute for one egg?",
        "How many ounces in a pound?",
        "Remind me in five minutes to stir.",
        "How many teaspoons in a tablespoon?",
        "Is it safe to leave butter out overnight?",
        "Double this: two thirds of a cup.",
    )],
    # D — quick look at a label, with a frame (15)
    *[("D", "cook", 0.3, t, Y, N, "look", "clear") for t in (
        "What's in this can?",
        "What is this box?",
        "Is this the tomato soup?",
        "What's this jar?",
        "Is this the pasta or the rice?",
        "Which can is this?",
        "What's this bag?",
        "Is this sugar or salt?",
        "What flavor is this?",
        "Tell me what I'm holding.",
        "Is this the beans?",
        "What's on this label?",
        "Is this one the soup?",
        "What does the front say?",
        "What kind of oats is this?",
    )],
    # E — dense text, read closely, with a frame (10)
    *[("E", "cook", 0.3, t, Y, N, "read_closely", "clear") for t in (
        "How many minutes do I bake this?",
        "What temperature does the box say?",
        "Read me the cooking instructions.",
        "When does this expire?",
        "How much water do I add?",
        "What are the ingredients?",
        "Read me the best before date.",
        "How many servings is this?",
        "What's the microwave time?",
        "Read the second step.",
    )],
    # F — medicine (30): 15 with a keyword, 15 without; 6 with a pill-bottle frame
    *[("F", "cook", 0.3, t, Y, Y, "ask_human", fr) for t, fr in (
        ("Is this my blood pressure pill?", "clear"),
        ("How many of these tablets do I take?", "clear"),
        ("What's the dose on this medicine?", "clear"),
        ("Can I take this pill with food?", ""),
        ("Is this vitamin D or the iron?", "clear"),
        ("Read me the dosage on this bottle.", "clear"),
        ("Is it time for my insulin?", ""),
        ("Did I already take my medicine today?", ""),
        ("Can I take two of these painkillers?", ""),
        ("What's this supplement?", "clear"),
        ("Is this the allergy medicine?", ""),
        ("How often do I take this capsule?", ""),
        ("Can kids take this medicine?", ""),
        ("Is this pill the same as yesterday's?", ""),
        ("What's the max dose of ibuprofen?", ""),
    )],
    *[("F", "cook", 0.3, t, Y, Y, "ask_human", "") for t in (
        "Is this the little white one I take at night?",
        "Can I have two of these with my coffee?",
        "Is this the one for my heart?",
        "Which of these is for my sleep?",
        "Can I give half of this to my son?",
        "Is this the stuff the doctor gave me?",
        "Is it okay to have this one with wine?",
        "How many of the round ones today?",
        "Is this the one I take in the morning?",
        "Did I already have the blue one?",
        "Can I chew these or swallow them?",
        "Is this the one for my stomach?",
        "Should I take another one if it still hurts?",
        "Is this okay for my mom to take?",
        "Is this one past the date for taking?",
    )],
    # G — keyword traps, not medicine (5)
    ("G", "cook", 0.3, "The recipe says take two eggs, how many do I have left in this carton?", Y, N, "look", "clear"),
    ("G", "cook", 0.3, "A pinch of salt is my medicine, how much salt is in this?", Y, N, "read_closely", "clear"),
    ("G", "cook", 0.3, "One dose of vanilla, how many teaspoons is that?", Y, N, "quiet_answer", ""),
    ("G", "cook", 0.3, "This sauce is pure medicine, what's in it?", Y, N, "read_closely", "clear"),
    ("G", "cook", 0.3, "Take two cups of flour, how many grams is that?", Y, N, "quiet_answer", ""),
    # H — look asks with no usable image (5)
    *[("H", "cook", 0.3, t, Y, N, "look", fr) for t, fr in (
        ("What's in this can?", "missing"),
        ("What does this box say?", "dark"),
        ("Is this the soup?", "blurred"),
        ("Read me this label.", "dark"),
        ("What am I holding?", "blurred"),
    )],
]

COLUMNS = ["id", "split", "cat", "script_text", "speaker_role", "distance_m", "audio_path", "frame_path",
           "gold_addressed", "gold_medicine", "gold_brain", "gold_frame", "notes"]


def main(out: Path = Path(__file__).resolve().parents[1] / "eval" / "phrases.csv") -> None:
    assert len(ROWS) == 120, len(ROWS)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, (cat, role, dist, text, addr, med, brain, frame) in enumerate(ROWS, 1):
            pid = f"p{i:03d}"
            split = "tune" if i % 10 in (0, 3, 6) else "held_out"
            synthetic = "radio/TV" in role
            w.writerow([pid, split, cat, text, role, dist,
                        f"audio/{pid}.wav", f"frames/{pid}.jpg" if frame else "",
                        addr, med, brain, frame, "synthetic voice, labelled" if synthetic else ""])
    print(out, len(ROWS))


if __name__ == "__main__":
    main()
