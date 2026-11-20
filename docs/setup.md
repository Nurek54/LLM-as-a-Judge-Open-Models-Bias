# Setup i uruchamianie

Wszystko idzie przez **llama.cpp**: `llama-quantize` robi drabinę kwantyzacji, `llama-server`
serwuje sędziego. Python tylko wysyła zapytania i liczy metryki.

## 1. llama.cpp (każda maszyna, TEN SAM build)

1. Wejdźcie na https://github.com/ggml-org/llama.cpp/releases i wybierzcie JEDEN release
   (`bNNNN`). Wpiszcie jego numer w `configs/models.yaml` → `llama_cpp.build`.
2. Pobierzcie paczkę dla swojej maszyny z tego samego release'u:
   - Windows + RTX 5080: `llama-bNNNN-bin-win-cuda-12.x-x64.zip` oraz `cudart-llama-bin-win-cuda-12.x-x64.zip`
     (rozpakować do jednego katalogu),
   - macOS M1: `llama-bNNNN-bin-macos-arm64.zip`.
3. W `.env` ustawcie `LLAMA_CPP_DIR` na katalog z `llama-server` / `llama-quantize`.
4. Test: `<LLAMA_CPP_DIR>/llama-server --version`.

## 2. Python

```powershell
# Windows (PowerShell), w katalogu repo
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # uzupełnić LLAMA_CPP_DIR i HF_TOKEN
```
```bash
# macOS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Modele gated (Bielik, Gemma, Llama): najpierw zaakceptować warunki na stronie modelu na HF,
potem `huggingface-cli login` albo `HF_TOKEN` w `.env`.

## 3. Dane

```bash
python scripts/prepare_data.py          # llmbar, judgebench, evalbiasbench, mtbench_human
python scripts/translate_llmbar.py      # LLMBar-PL do P3 (wymaga TRANSLATE_* w .env)
```
Po tłumaczeniu: otworzyć `data/processed/llmbar_pl_weryfikacja.csv` i ręcznie ocenić próbę
(kolumna `ok`). Wynik tej weryfikacji idzie do raportu.

## 4. Modele

```bash
python scripts/models.py check qwen3.5-9b        # czy wzorzec BF16 pasuje do plików w repo
python scripts/models.py download qwen3.5-9b     # BF16 -> models/qwen3.5-9b/
python scripts/models.py quantize qwen3.5-9b --quants Q8_0 Q6_K Q4_K_M Q3_K_M
python scripts/models.py info --sha              # rozmiar, bity/wagę, SHA-256 -> configs/models_info.json
```
Repo oznaczone w configu `verify: true` najpierw `check`. Jeśli wzorzec nie pasuje,
poprawcie `source.pattern` (albo `source.repo`) w `configs/models.yaml`.

Miejsce na dysku: BF16 ≈ 2 GB na 1B parametrów (wszystkie 8 modeli ≈ 150 GB),
Q4_K_M ≈ 0,6 GB/1B. Pliki Q4 do P1 i P3 wystarczy zrobić raz i skopiować na inne maszyny
(SHA-256 z `info --sha` potwierdza, że to te same pliki).

## 5. Test dymny, potem eksperymenty

```bash
python scripts/run.py check qwen3.5-4b Q4_K_M            # surowa odpowiedź bez gramatyki
python scripts/run.py p1 --model qwen3.5-4b --limit 20   # 20 par na zbiór
```
`check` pokazuje, co model generuje bez ograniczeń i jaka część prawdopodobieństwa
przypada na litery A/B (`masa_A+B`). Jeśli masa jest niska albo model zaczyna od bloku
myślenia, trzeba poprawić `chat_template_kwargs` / `assistant_prefill` dla tego modelu.
**Zróbcie `check` dla każdego modelu przed pełnym przebiegiem.**

Pełne przebiegi (każdy eksperyment w całości na jednej maszynie):
```bash
python scripts/run.py p1     # PC 5080
python scripts/run.py p2     # laptop 5080
python scripts/run.py p3     # MacBook M1
```
Przebieg można przerwać i wznowić tą samą komendą: zapisane wywołania są pomijane.

## 6. BF16 z offloadem (P2)

Qwen3.5-9B i Ministral-3-8B w BF16 nie mieszczą się w 16 GB VRAM. `ngl_by_quant.BF16`
w configu to liczba warstw na GPU. Dobierzcie ją raz: zacznijcie od wartości z configu,
odpalcie `run.py check <model> BF16` i patrzcie na zużycie VRAM (Menedżer zadań → GPU).
Jeśli serwer się wywala z OOM, zmniejszcie o 2–4 warstwy. Liczba warstw nie wpływa na
wynik, tylko na czas. Jeśli BF16 będzie nieakceptowalnie wolne: `fallback_models` w P2
(Qwen3.5-4B, mieści się cały w VRAM).

## 7. Analiza

```bash
python scripts/analyze.py p1 --ci accuracy pc
python scripts/analyze.py p2 --mcnemar qwen3.5-9b BF16 Q4_K_M --dataset llmbar
python scripts/analyze.py p3 --mcnemar-lang bielik-11b Q4_K_M
```
Tabela trafia do `results/<exp>/summary.csv`. Opis metryk: `src/judge/metrics.py`.
Uwaga: w EvalBiasBench gold zawsze wskazuje odpowiedź 1, więc κ dla tego zbioru nie ma
sensu; patrzymy na accuracy i position bias.

## Zasady, które pilnują porównywalności

- Jeden build llama.cpp na wszystkich maszynach; nie aktualizować w trakcie.
- Wszystkie kwantyzacje robimy sami z BF16, bez imatrix (gotowe GGUF od różnych autorów
  bywają robione z imatrix i różnymi ustawieniami).
- Nie zmieniać `ctx`, `parallel`, `extra_args`, promptu ani gramatyki w trakcie eksperymentu.
- KV cache zostaje domyślny (F16), bez `--cache-type-k/v`.
- Thinking wyłączony (`enable_thinking: false` / brak tokenu think), sprawdzone `check`.
