# P1: Position bias a rodzina i rozmiar modelu

**Pytanie:** Jak silny jest position bias zależnie od rodziny i rozmiaru modelu?

**Zmienne:** rodzina modelu (np. Llama, Qwen, Mistral, Gemma, Bielik) i rozmiar (7-8B, 12-14B).
Kwantyzacja i język bez zmian.

**Procedura (do uzgodnienia):** każda para oceniana dwa razy, w kolejności A-B i B-A.

**Metryki:** spójność werdyktu po zamianie kolejności (position consistency), odsetek wyborów
pierwszej pozycji, zgodność z oceną ludzi (Cohen's kappa).

**Dane:** MT-Bench human judgments, LLMBar.

**Do ustalenia z prowadzącym:** lista modeli, liczba par, test statystyczny.
