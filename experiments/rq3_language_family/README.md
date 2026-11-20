# P3: Język ocenianych odpowiedzi

**Pytanie:** Jak język ocenianych odpowiedzi (polski vs angielski) wpływa na trafność
i spójność werdyktów sędziów LLM?

**Hipoteza H3:** sędziowie są mniej trafni i mniej spójni po polsku niż po angielsku,
a różnica jest mniejsza dla modelu specjalizowanego w polskim (Bielik).

**Zmienna niezależna:** język pary (EN: LLMBar, PL: LLMBar-PL), prompt sędziego w tym samym
języku co dane. Model i kwantyzacja (Q4_K_M) stałe w obrębie porównania.

**Sędziowie:** qwen3.5-9b, gemma-4-12b, ministral-3-8b, bielik-11b (configs/experiments.yaml).

**Metryki:** accuracy, κ, position consistency, soft position bias, format_mass; test McNemara
na tych samych parach EN vs PL (`scripts/analyze.py p3 --mcnemar-lang <model> Q4_K_M`).

**Dane:** LLMBar (419 par) i jego tłumaczenie (`scripts/translate_llmbar.py`), etykiety gold
zachowane. Jakość tłumaczenia weryfikowana ręcznie na losowej próbie 40 par.
Dodatkowo PLLuM-Align jako zbiór natywnie polski (loader do dopisania, gdy ustalimy format).

**Maszyna:** MacBook M1 (cały eksperyment).

Wątek „preferencji własnej rodziny” odłożony: wymagałby osobnego eksperymentu, a każde
pytanie ma mieć jeden.
