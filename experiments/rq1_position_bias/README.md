# P1: Position bias a rodzina i rozmiar modelu

**Pytanie:** Jak siła position bias sędziego LLM zależy od rodziny i rozmiaru modelu?

**Hipoteza H1:** w obrębie rodziny mniejsze modele mają silniejszy position bias.

**Zmienne:** rodzina (Qwen3.5, Gemma 4, Ministral 3, Bielik v3; opcjonalnie Llama 3.1)
i rozmiar (mały ~4–8B, większy ~9–14B). Kwantyzacja stała: Q4_K_M. Język: EN.

**Procedura:** każda para oceniana w kolejności A-B i B-A, temperature=0, odpowiedź wymuszona
gramatyką (A/B), zapisywane P(A) z logprobs.

**Metryki:** position consistency, first_rate / pb, soft position bias, accuracy i κ względem
ludzi (MT-Bench) lub gold (LLMBar), bootstrap CI po parach.

**Dane:** MT-Bench human judgments (tura 1, większość głosów, bez remisów), LLMBar (419 par).

**Maszyna:** PC RTX 5080 (cały eksperyment).
