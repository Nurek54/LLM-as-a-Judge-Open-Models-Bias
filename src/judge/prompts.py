"""Prompt sędziego (pairwise). Ta sama treść dla wszystkich modeli i poziomów kwantyzacji.

Sędzia ma zwrócić wyłącznie literę A albo B; gramatyka w llama-server i tak wymusza ten
format, a prawdopodobieństwa obu liter odczytujemy z logprobs.
"""
from __future__ import annotations

SYSTEM = {
    "en": (
        "You are an impartial judge evaluating the quality of two AI assistant responses "
        "to the same user instruction. Choose the response that better follows the "
        "instruction and is more helpful, accurate and harmless. Do not let the order in "
        "which the responses are presented, their length or their formatting influence "
        "your decision. Answer with a single letter: A or B."
    ),
    "pl": (
        "Jesteś bezstronnym sędzią oceniającym jakość dwóch odpowiedzi asystenta AI na to "
        "samo polecenie użytkownika. Wybierz odpowiedź, która lepiej wykonuje polecenie "
        "oraz jest bardziej pomocna, poprawna i bezpieczna. Kolejność prezentacji "
        "odpowiedzi, ich długość ani formatowanie nie mogą wpływać na Twoją decyzję. "
        "Odpowiedz jedną literą: A albo B."
    ),
}

USER = {
    "en": (
        "[User instruction]\n{instruction}\n\n"
        "[Response A]\n{a}\n[End of Response A]\n\n"
        "[Response B]\n{b}\n[End of Response B]\n\n"
        "Which response is better? Answer with A or B only."
    ),
    "pl": (
        "[Polecenie użytkownika]\n{instruction}\n\n"
        "[Odpowiedź A]\n{a}\n[Koniec odpowiedzi A]\n\n"
        "[Odpowiedź B]\n{b}\n[Koniec odpowiedzi B]\n\n"
        "Która odpowiedź jest lepsza? Odpowiedz tylko A albo B."
    ),
}

# Gramatyka GBNF dla llama-server: dokładnie jeden znak A albo B.
VERDICT_GRAMMAR = 'root ::= "A" | "B"'


def build_messages(pair: dict, order: str, lang: str, prefill: str | None = None) -> list[dict]:
    """order '12': response_1 jako A; order '21': response_1 jako B."""
    a, b = ((pair["response_1"], pair["response_2"]) if order == "12"
            else (pair["response_2"], pair["response_1"]))
    msgs = [
        {"role": "system", "content": SYSTEM[lang]},
        {"role": "user", "content": USER[lang].format(instruction=pair["instruction"], a=a, b=b)},
    ]
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
    return msgs


def slot_to_response(slot: str, order: str) -> int:
    """Zamienia wybraną pozycję (A/B) na numer oryginalnej odpowiedzi (1/2)."""
    first = 1 if order == "12" else 2
    return first if slot == "A" else 3 - first
