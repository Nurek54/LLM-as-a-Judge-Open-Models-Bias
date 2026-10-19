# P2: Wpływ kwantyzacji na stronniczość sędziego

**Pytanie:** Czy kwantyzacja (FP16, Q8, Q4) zwiększa stronniczość sędziego?

**Zmienne:** poziom kwantyzacji tego samego modelu. Model, prompt i dane bez zmian.

**Metryki:** zmiana position consistency i length bias między poziomami, trafność na parach
z poprawną odpowiedzią.

**Dane:** JudgeBench, EvalBiasBench, LLMBar.

**Uwaga sprzętowa:** FP16 dla 7-8B wymaga ok. 16 GB VRAM. Jeśli brak, punktem odniesienia jest Q8.

**Luka badawcza:** wpływ kwantyzacji na bias jest badany poza rolą sędziego, ale nie w niej.
