"""The dimensions Ledger reports, and the anchor phrases that define them.

These are **language signals**, not diagnoses and not clinical instrument scores.
Nothing here is PHQ-9, GAD-7 or any other validated instrument; plan.md risk R-2
keeps those out until their licences are checked, and the naming below is
deliberately about *what the person wrote* rather than about what they have.

The anchors are original text written for this project, so nothing in this file
carries a third-party licence. That matters for hackathon rule 2 (attribution)
and it keeps the zero-shot head reproducible from the repository alone.
"""

HEAD_VERSION = "anchor_v0"

#: Stable order. The ONNX graph's output columns follow this tuple exactly.
DIMENSIONS = (
    "low_mood",
    "anxiety",
    "sleep_disruption",
    "social_withdrawal",
    "activation",
)

#: Human-facing labels. Phrased as observations about the writing.
DIMENSION_LABELS = {
    "low_mood": "Low-mood language",
    "anxiety": "Anxious / worried language",
    "sleep_disruption": "Sleep-disruption language",
    "social_withdrawal": "Withdrawal-from-others language",
    "activation": "Energy and activity language",
}

#: Each dimension is defined by a positive and a negative pole. The head row is
#: the difference of the two pole centroids, so a score is a *contrast* and not
#: an absolute magnitude — which is what makes the zero point interpretable
#: without any training data.
ANCHORS = {
    "low_mood": {
        "positive": [
            "Today felt heavy and grey from the moment I woke up.",
            "I could not find anything to look forward to.",
            "Everything took more effort than it should have.",
            "I felt flat, like the colour had gone out of the day.",
            "I kept thinking I was letting people down.",
        ],
        "negative": [
            "Today felt light and I was glad to be in it.",
            "I was looking forward to the evening all afternoon.",
            "Things came easily and I enjoyed most of them.",
            "I felt steady and content for most of the day.",
            "I was pleased with how I handled the afternoon.",
        ],
    },
    "anxiety": {
        "positive": [
            "My chest was tight all morning and I could not settle.",
            "I kept rehearsing the conversation over and over.",
            "I was braced for something to go wrong.",
            "My thoughts would not slow down at all today.",
            "I checked the message four times before sending it.",
        ],
        "negative": [
            "I felt calm and unhurried for most of the day.",
            "I said what I meant without going over it afterwards.",
            "Nothing felt urgent and that was fine.",
            "My mind was quiet while I worked.",
            "I sent it once and did not think about it again.",
        ],
    },
    "sleep_disruption": {
        "positive": [
            "I was awake until four and then up again at six.",
            "I kept waking through the night and could not get back down.",
            "I lay there for hours with my eyes open.",
            "I slept badly and dragged through the whole day.",
            "I woke long before the alarm and could not sleep again.",
        ],
        "negative": [
            "I slept right through and woke up rested.",
            "I went down easily and did not wake once.",
            "Eight solid hours and I felt it all day.",
            "I woke naturally just before the alarm and felt fine.",
            "Sleep has been steady all week.",
        ],
    },
    "social_withdrawal": {
        "positive": [
            "I cancelled again and stayed in on my own.",
            "I let the calls go to voicemail all weekend.",
            "I did not want to see anyone or explain myself.",
            "I have not replied to anybody in days.",
            "I ate alone and was relieved not to talk.",
        ],
        "negative": [
            "I met a friend for lunch and stayed longer than planned.",
            "I called my sister and we talked for an hour.",
            "I went along even though I nearly did not, and was glad.",
            "There were people around all evening and it was easy.",
            "I replied to everyone I had been meaning to.",
        ],
    },
    "activation": {
        "positive": [
            "I got out early and walked for an hour before work.",
            "I cleared the whole list and started on tomorrow's.",
            "I cooked properly and cleaned the kitchen afterwards.",
            "I had energy left over at the end of the day.",
            "I started the thing I have been putting off.",
        ],
        "negative": [
            "I did not leave the flat or get dressed.",
            "I meant to start and never did.",
            "I sat on the sofa for most of the day.",
            "I had nothing left after the morning.",
            "The list is exactly where it was yesterday.",
        ],
    },
}

assert tuple(ANCHORS) == DIMENSIONS, "ANCHORS must be in DIMENSIONS order"
