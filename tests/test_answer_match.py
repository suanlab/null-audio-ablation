"""Reading a closed-vocabulary answer out of a model's free text.

Every failure mode here was a live risk for the MUSIC-AVQA anchor: the vocabulary contains
`no` and `nine`, `one` and `zero`, and multi-word entries like `acoustic guitar`, and the
models wrap answers in prose. A misread here does not surface as an error -- it silently
scores an item wrong, and since Delta_A is a paired difference, a systematic misread would
bias every cell in the same direction.
"""

from __future__ import annotations

import pytest

from videollm.data.answer_match import match_answer, normalise

VOCAB = (
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon", "cello", "clarinet",
    "congas", "drum", "eight", "electric_bass", "erhu", "five", "flute", "four", "guzheng",
    "indoor", "left", "middle", "more_than_ten", "nine", "no", "one", "outdoor", "piano",
    "pipa", "right", "saxophone", "seven", "simultaneously", "six", "suona", "three",
    "trumpet", "tuba", "two", "ukulele", "violin", "xylophone", "yes", "zero",
)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("piano", "piano"),
        ("  Piano.  ", "piano"),
        ("acoustic guitar", "acoustic_guitar"),
        ("acoustic_guitar", "acoustic_guitar"),
        ("The answer is acoustic guitar.", "acoustic_guitar"),
        ("Answer: two", "two"),
        ("more than ten", "more_than_ten"),
    ],
)
def test_reads_the_plain_answer(reply: str, expected: str) -> None:
    assert match_answer(reply, VOCAB) == expected


def test_leading_yes_no_is_not_overridden_by_a_later_instrument() -> None:
    """Regression: a longest-match reader returns `flute` here and inverts the answer."""
    assert match_answer("No, the flute is not playing.", VOCAB) == "no"
    assert match_answer("Yes, the violin and the piano both sound.", VOCAB) == "yes"


def test_word_boundaries_are_respected() -> None:
    """`one` must not fire inside `someone`, nor `no` inside `nothing`."""
    assert match_answer("someone is offstage", VOCAB) == ""
    assert match_answer("nothing is audible", VOCAB) == ""


def test_similar_number_words_are_not_confused() -> None:
    assert match_answer("nine", VOCAB) == "nine"
    assert match_answer("no", VOCAB) == "no"
    assert match_answer("zero instruments", VOCAB) == "zero"


def test_unreadable_replies_return_empty_rather_than_guessing() -> None:
    """An empty string is recorded as unparseable; a guess would be scored as an answer."""
    assert match_answer("", VOCAB) == ""
    assert match_answer("I cannot tell from the video.", VOCAB) == ""
    assert match_answer("!!!", VOCAB) == ""


def test_earliest_match_wins_when_several_terms_appear() -> None:
    assert match_answer("left, though the piano is on the right", VOCAB) == "left"


def test_normalise_is_stable_across_underscore_and_punctuation() -> None:
    assert normalise("Acoustic_Guitar!") == "acoustic guitar"
    assert normalise("  more   than_ten  ") == "more than ten"


def test_empty_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        match_answer("piano", ())


def test_option_recitation_is_not_scored_as_the_first_option() -> None:
    """Regression: a model that echoes the candidate list must not be read as answering.

    video-SALMONN v1 is a captioning model, not tuned for closed-vocabulary selection, and
    produces replies that name most of the option set. Earliest-match would score those as
    whichever option happens to be alphabetically first.
    """
    from videollm.data.answer_match import is_recitation

    recite = (
        "There are multiple types of musical instruments sounding in the video, including "
        "an accordion, acoustic guitar, bagpipe, banjo, bassoon, cello, clarinet, congas, "
        "drum, flute, guzheng, piano, violin, and xylophone."
    )
    assert is_recitation(recite, VOCAB)
    assert match_answer(recite, VOCAB) == "accordion"  # what the naive read would give

    assert not is_recitation("piano", VOCAB)
    assert not is_recitation("No, the flute is not playing.", VOCAB)
    assert not is_recitation("Either the violin or the cello.", VOCAB)
