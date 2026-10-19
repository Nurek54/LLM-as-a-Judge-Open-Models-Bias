# LLM-as-a-Judge: Bias in Open-Weight Models

Projekt z kursu *Metody przetwarzania języka naturalnego oraz wyszukiwania*, Politechnika Wrocławska.
Temat: **LLM-as-a-judge - stronniczość otwartych modeli**.

**Zespół (grupa 3):** Miłosz Bedryło, Grzegorz Sienkiewicz, Kamil Sromek

## Problem

Badamy, czy otwarte modele językowe (7-14B, kwantyzacja Q4/Q8) uruchamiane lokalnie w roli sędziego
oceniają odpowiedzi stronniczo: według kolejności, języka lub rodziny modelu, a nie jakości.
Jako punkt odniesienia używamy 1-2 modeli zamkniętych.

## Pytania badawcze

Każde pytanie ma osobny eksperyment w `experiments/`.

| | Pytanie | Eksperyment |
|---|---|---|
| P1 | Jak silny jest position bias zależnie od rodziny i rozmiaru modelu? | `experiments/rq1_position_bias/` |
| P2 | Czy kwantyzacja (FP16, Q8, Q4) zwiększa stronniczość sędziego? | `experiments/rq2_quantization/` |
| P3 | Czy sędziowie oceniają inaczej po polsku niż po angielsku i czy faworyzują własną rodzinę modeli? | `experiments/rq3_language_family/` |

## Dane

| Zbiór | Do czego | Źródło |
|---|---|---|
| MT-Bench human judgments | position bias, zgodność z ludźmi | [HF](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) |
| Chatbot Arena conversations | duża próba par z głosami ludzi | [HF](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) |
| LLMBar | trudne pary, gorsza odpowiedź wygląda lepiej | [GitHub](https://github.com/princeton-nlp/LLMBar) |
| JudgeBench | pary z obiektywnie poprawną odpowiedzią | [GitHub](https://github.com/ScalerLab/JudgeBench) |
| RewardBench | pary preferencji z podziałem na kategorie | [HF](https://huggingface.co/datasets/allenai/reward-bench) |
| EvalBiasBench (OffsetBias) | przypadki na konkretne typy biasu | [GitHub](https://github.com/ncsoft/offsetbias) |

## Struktura repozytorium

```
configs/        konfiguracje sędziów, promptów i eksperymentów
data/raw/       pobrane zbiory (poza gitem)
data/processed/ przygotowane pary do oceny (poza gitem)
src/            kod wspólny: ładowanie danych, klient sędziego, metryki
experiments/    po jednym katalogu na pytanie badawcze
notebooks/      analiza i wykresy
results/        surowe werdykty sędziów (poza gitem)
models/         lokalne pliki GGUF (poza gitem)
report/         sprawozdanie LNCS, osobno dla każdego etapu
```

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # uzupełnić klucze
```

Do P2 z sędzią w FP16 potrzebne są dodatkowo `requirements-gpu.txt` oraz GPU z ok. 16 GB VRAM (model 7-8B).

Lokalne modele uruchamiamy przez [Ollama](https://ollama.com) albo serwer llama.cpp.
Oba wystawiają API zgodne z OpenAI, więc wszyscy sędziowie (lokalni i zamknięci) mają ten sam klient.

## Wymagania kursu

- Grupy 3-osobowe, projekt z aspektem badawczym.
- 3 pytania badawcze sprawdzające różne właściwości, do każdego jeden eksperyment i szersza odpowiedź.
- Protokół eksperymentalny (metryki, walidacja, test statystyczny) uzgadniany z prowadzącym.
- Sprawozdanie w formie artykułu naukowego, szablon LNCS, bibliografia z cytowaniami.
- Etap I: ponad 15 przeczytanych i zacytowanych prac.
- Osobna sekcja o wykorzystaniu narzędzi AI (zakres + zapytania i odpowiedzi). Wygenerowany tekst w sprawozdaniu jest zabroniony.
- Każdy etap konsultowany przed oddaniem, przesyłany na Eportal. Spóźnienie: -10 pkt za etap.
- Na ostatnich zajęciach omówienie projektu w pełnym składzie.

## Harmonogram

| Etap | Zakres | Termin | Status |
|---|---|---|---|
| I | Studium literaturowe: wstęp, przegląd literatury | 04.11.2026 | w trakcie |
| II | Projekt i implementacja: opis metod, plan eksperymentów | 09.12.2026 | |
| III | Ewaluacja: analiza wyników, wnioski, omówienie | 27.01.2027 | |

Termin poprawkowy (maks. 3.0): 18.02.2027.

## Punktacja

**Projekt (15 pkt):** dobór danych 2, implementacja 3, pytania badawcze 5, eksperymenty 4, prezentacja wyników 1.
**Sprawozdanie (15 pkt):** wstęp 1, przegląd literatury 2, opis metod 3, plan eksperymentów 3, analiza wyników 4, wnioski 2.
