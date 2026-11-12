# F1nalyze: classifying Formula 1 race results

An ML project on the data from the F1nalyze Formula 1 Datathon (Kaggle).
The task is to predict which of four classes a driver finishes in: **win (1) / podium (2–3) / points (4–10) / rest (>10)**.
The target is `positionOrder`. The main metric is macro-F1, with balanced accuracy as a secondary one.

## Key finding

**No ML model beat a simple rule based on the starting grid position (`grid`).**

| Model | Val macro-F1 | Test macro-F1 |
|---|---:|---:|
| **Grid rule** (grid 1 → win, 2–3 → podium, 4–10 → points, >10 → rest) | **0.6006** | **0.5532** |
| Hierarchical (top-10 vs rest → 3 classes) | 0.5628 | 0.5411 |
| Logistic regression, no class weights | 0.5310 | 0.5357 |
| LightGBM, balanced | 0.5461 | 0.5335 |
| LightGBM, balanced + class weights tuned on val | 0.5705 | 0.5187 |
| Random Forest, no class weights | 0.5304 | 0.5167 |
| LightGBM, trained on 2000–2013 only | 0.5472 | 0.4868 |

Tuning class weights on the validation set improved the val score but lowered the test score: the model overfit.
The hardest class is podium; the best boosting model most often confuses it with win.

## Data pitfalls (worth reading before using these CSVs)

1. **Broken join.** The raw files are a Cartesian product: every result (`resultId`) has championship-standings
   snapshots from many rounds attached (~128× inflation in train). Fix: keep the row where `raceId_y == racerId`;
   when it is missing (~2% of results), take the nearest earlier snapshot and blank the post-race columns.
2. **`racerId` is a race ID, not a driver ID.** The driver is `driverId`.
3. **Data leakage.** Columns that are only known after the race (`points`, `position_x`, `rank`, `laps`, times,
   `fastestLap*`, `max_speed`, `status`) are excluded from the features. `points_y` and `wins` are only used lagged,
   and rolling features are computed with `shift(1)`.
4. **Time-based split.** Train 1950–2013, val 2013–2018, test 2018–2023. Cross-validation uses `TimeSeriesSplit` only.

## Repository layout

| File | Description |
|---|---|
| `00_dataset_overview.ipynb` | Dataset presentation: variable dictionary, missingness, target, EDA charts, strengths/weaknesses |
| `gen_dataset_overview.py` | **Generator** of notebook 00. Make changes here, not in the `.ipynb` |
| `01_phase1_data_profiling.ipynb` | Data profiling, discovering the broken join, class distribution |
| `02_modeling_classification.ipynb` | Data repair, EDA, feature engineering, models, evaluation |
| `gen_notebook.py` | **Generator** of notebook 02. Make changes here, not in the `.ipynb` |
| `reports/` | Report and presentation (PDF) |
| `data/` | Kaggle data (not included in the repo) |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download train.csv, validation.csv, test.csv and sample_submission.csv from Kaggle into data/
python gen_notebook.py          # rebuilds 02_modeling_classification.ipynb
jupyter lab
```

`train.csv` is ~1.1 GB and is read in chunks. Random seed: `SEED = 42`.

## Limitations and next steps

- The predictions file (`data/predictions_target_A.csv`) contains the 4-level class, while `sample_submission.csv`
  expects a numeric `position`. It is therefore not a valid competition submission.
- Ideas: predict the change of position relative to the grid instead of the position itself, a separate P(DNF) or
  podium model, qualifying data from external sources (it is empty in this dataset).
