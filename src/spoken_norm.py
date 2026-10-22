"""Normalise spoken maths text for WER: Whisper writes "5x² = 10", the verbaliser "five x squared equals ten"."""

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SYMBOLS = {"=": " equals ", "+": " plus ", "±": " plus or minus ", "×": " times ", "·": " times ",
            "/": " over ", "≤": " less than or equal to ", "≥": " greater than or equal to ",
            "<": " less than ", ">": " greater than ", "²": " squared ", "³": " cubed ", "√": " square root of "}
_FILLER = {"is", "the", "a", "an", "of", "quantity", "by"}   # optional words, dropped on both sides


def number_words(n):
    """Cardinal words for 0 <= n < 1,000,000."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))


def normalise_spoken(text):
    """Lowercase word string with digits and symbols spelled out and filler words dropped."""
    s = text.lower().replace("−", "-")
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", s)   # "twenty-one"
    # a minus with nothing (or an operator/relation) before it is a sign: "-4x", "= -3", "plus -2"
    s = re.sub(r"(^|[=<>(,≤≥+]|\b(?:equals|than|to|or|plus|minus|times|over)\b)\s*-\s*(?=[\dx(])",
               r"\1 negative ", s)
    s = s.replace("-", " minus ")
    for sym, word in _SYMBOLS.items():
        s = s.replace(sym, word)
    s = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} " if len(m.group()) < 7 else m.group(), s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(w for w in s.split() if w not in _FILLER)
