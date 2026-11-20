# P2: Wpływ kwantyzacji na sędziego

**Pytanie:** W jakim stopniu kwantyzacja wag (BF16 → Q8 → Q4) wpływa na trafność oraz
stronniczość (position bias, preferencja dłuższych odpowiedzi) modelu pełniącego rolę sędziego?

**Hipoteza H2:** wraz ze spadkiem precyzji spada trafność i rośnie stronniczość; efekt
wyraźny dopiero poniżej Q8.

**Zmienna niezależna:** poziom kwantyzacji tego samego modelu: BF16, Q8_0, Q6_K, Q4_K_M, Q3_K_M.
Wszystkie poziomy z jednego pliku BF16, jednym buildem `llama-quantize`, bez imatrix.
Prompt, dane, silnik i ustawienia serwera bez zmian.

**Modele:** qwen3.5-9b, ministral-3-8b (zapasowo qwen3.5-4b, mieści się w VRAM w BF16).

**Metryki:** accuracy, κ, position consistency, first_rate / pb, soft position bias (z logprobs),
longer_rate i accuracy przy gold dłuższej/krótszej, format_mass, RS (3 powtórzenia przy T=0.7).
Bity na wagę z `scripts/models.py info`. Test McNemara między poziomami
(`scripts/analyze.py p2 --mcnemar <model> BF16 <quant> --dataset <zbiór>`).

**Dane:** LLMBar, JudgeBench (350 par, odpowiedzi GPT-4o), EvalBiasBench (80 par).

**Maszyna:** laptop RTX 5080 (cały eksperyment). BF16 z częściowym offloadem do RAM.
