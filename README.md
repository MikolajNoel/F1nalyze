# F1nalyze: klasyfikacja wyników wyścigów F1

Projekt ML na danych z F1nalyze Formula 1 Datathon (Kaggle).
Przewiduję, w której z czterech klas skończy kierowca: **wygrana (1) / podium (2–3) / punkty (4–10) / reszta (>10)**.
Target to `positionOrder`, metryka główna to macro-F1, pomocnicza to balanced accuracy.

## Najważniejszy wniosek

**Żaden model ML nie pobił prostej reguły opartej na polu startowym (`grid`).**

| Model | Val macro-F1 | Test macro-F1 |
|---|---:|---:|
| **Reguła grid** (grid 1 → wygrana, 2–3 → podium, 4–10 → punkty, >10 → reszta) | **0.6006** | **0.5532** |
| Hierarchiczny (top10 vs reszta → 3 klasy) | 0.5628 | 0.5411 |
| Regresja logistyczna bez wag | 0.5310 | 0.5357 |
| LightGBM balanced | 0.5461 | 0.5335 |
| LightGBM balanced + wagi klas strojone na val | 0.5705 | 0.5187 |
| Random Forest bez wag | 0.5304 | 0.5167 |
| LightGBM, train tylko 2000–2013 | 0.5472 | 0.4868 |

Strojenie wag klas na zbiorze walidacyjnym poprawiło wynik na val, a pogorszyło na teście, czyli model się przeuczył.
Najtrudniejsza klasa to podium, mylona z wygraną i z punktami.

## Pułapki w danych (warto przeczytać przed użyciem tych CSV)

1. **Zepsuty join.** Surowe pliki to iloczyn kartezjański: każdy wynik (`resultId`) ma doklejone migawki
   klasyfikacji generalnej z wielu rund (inflacja ~128× w train). Naprawa: zostaw wiersz, w którym `raceId_y == racerId`,
   a gdy go brakuje (~2% wyników), weź najbliższą wcześniejszą migawkę i wyzeruj kolumny po-wyścigowe.
2. **`racerId` to ID wyścigu, nie kierowcy.** Kierowca to `driverId`.
3. **Wyciek danych.** Kolumny znane dopiero po wyścigu (`points`, `position_x`, `rank`, `laps`, czasy, `fastestLap*`,
   `max_speed`, `status`) są wykluczone z cech. `points_y` i `wins` są używane tylko po zlagowaniu, a cechy rolling liczone są z `shift(1)`.
4. **Podział czasowy.** Train 1950–2013, val 2013–2018, test 2018–2023. Walidacja krzyżowa tylko `TimeSeriesSplit`.

## Struktura repo

| Plik | Opis |
|---|---|
| `01_Faza1_profilowanie_danych.ipynb` | Profilowanie danych, wykrycie zepsutego joina, rozkład klas |
| `02_modelowanie_klasyfikacja.ipynb` | Naprawa danych, EDA, feature engineering, modele, ocena |
| `gen_notebook.py` | **Generator** notebooka 02. Zmiany wprowadzaj tutaj, nie w `.ipynb` |
| `reports/` | Raport i prezentacja (PDF) |
| `data/` | Dane z Kaggle (nie są w repo) |

## Jak uruchomić

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pobierz train.csv, validation.csv, test.csv, sample_submission.csv z Kaggle do data/
python gen_notebook.py          # odtwarza 02_modelowanie_klasyfikacja.ipynb
jupyter lab
```

`train.csv` ma ~1,1 GB i jest czytany chunkami. Ziarno losowe: `SEED = 42`.

## Ograniczenia i dalsze kierunki

- Plik z predykcjami (`data/predykcje_target_A.csv`) zawiera klasę 4-poziomową, a `sample_submission.csv` oczekuje liczbowego `position`.
  To nie jest więc gotowe zgłoszenie konkursowe.
- Pomysły: przewidywać zmianę pozycji względem gridu zamiast samej pozycji, osobny model P(DNF) albo podium,
  dane z kwalifikacji ze źródeł zewnętrznych (w tym zbiorze są puste).
