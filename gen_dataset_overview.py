#!/usr/bin/env python3
"""Generates 00_dataset_overview.ipynb — a presentation-ready overview of the F1nalyze dataset.

Make changes here, not in the .ipynb:  python gen_dataset_overview.py
"""
import json


def cc(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.strip("\n"), "outputs": [], "execution_count": None}


def mc(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.strip("\n")}


cells = [

# ════════════════════════════════════════════════════════════════════════════
mc("title", r"""
# 🏎️ F1nalyze — dataset presentation

**74 seasons of Formula 1 (1950–2023) in one dataset: what is in it, how to read it, and what to watch out for.**

| | |
|---|---|
| **Source** | Kaggle — *F1nalyze: Formula 1 Datathon* (historical F1 data, Ergast database structure) |
| **Files** | `train.csv` (1950–2013), `validation.csv` (2013–2018), `test.csv` (2018–2023), `sample_submission.csv` |
| **Unit of observation** | one **result**: driver × race (`resultId`) |
| **Project target** | `positionOrder` → 4 classes: win / podium / points / rest |

### Agenda
1. The dataset in numbers
2. Files and the time-based split
3. ⚠️ Pitfall #1: the broken join (Cartesian product) — and the fix
4. Variable dictionary — what every column means
5. Missing values — when and why
6. The target variable
7. Exploration: seasons, starting grid, retirements, legends, points, circuits, speed
8. Fun facts and records
9. Drift over time — why the past does not fully describe the future
10. ✅ Strengths and ❌ weaknesses of the dataset
11. Takeaways for modelling

> The notebook is self-contained: it reads the CSVs from `data/`, repairs them and writes a cache
> (`data/_cache_dataset_overview.pkl`), so subsequent runs take a few seconds.
"""),

cc("setup", r"""
import os, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 200)

DATA_DIR = Path(os.environ.get('F1_DATA_DIR', 'data'))
CACHE    = DATA_DIR / '_cache_dataset_overview.pkl'
NA_VALS  = [r'\N']
SPLITS   = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
POST_RACE_STANDINGS = ['points_y', 'wins', 'position', 'positionText_y']

# ── Chart styling: one consistent palette across the whole notebook ───────────
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID_C, AXIS_C = '#e1e0d9', '#c3c2b7'
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
         magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948', grey='#c3c2b7')
SPLIT_COLORS = {'train': C['blue'], 'validation': C['orange'], 'test': C['aqua']}
CLASSES      = ['win', 'podium', 'points', 'rest']
CLASS_LABELS = {'win': 'win (1)', 'podium': 'podium (2–3)', 'points': 'points (4–10)', 'rest': 'rest (>10)'}
CLASS_COLORS = {'win': C['yellow'], 'podium': C['blue'], 'points': C['aqua'], 'rest': C['grey']}
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', ['#f4f8fd', '#9ec5f4', '#3987e5', '#1c5cab', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a8f', '#b52a2a'])

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'axes.edgecolor': AXIS_C, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID_C, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.frameon': False, 'legend.fontsize': 9,
    'lines.linewidth': 2, 'patch.linewidth': 0,
})

def subtitle(fig_or_ax, text, y=None):
    # Subtitle under a chart title (grey, smaller font)
    if hasattr(fig_or_ax, 'suptitle'):
        fig_or_ax.text(0.01, y or 0.93, text, fontsize=9.5, color=INK2, transform=fig_or_ax.transFigure)
    else:
        fig_or_ax.set_title(fig_or_ax.get_title(loc='left'), loc='left', pad=24)
        fig_or_ax.text(0, 1.012, text, fontsize=9, color=INK2, transform=fig_or_ax.transAxes, va='bottom')

def fig_title(fig, title, sub=None):
    fig.suptitle(title, x=0.01, ha='left', fontsize=14, fontweight='bold', y=1.02 if sub else 0.99)
    if sub:
        fig.text(0.01, 0.955, sub, fontsize=10, color=INK2, ha='left')

def pct_axis(ax, axis='y'):
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

def fmt_int(x):
    return f'{int(x):,}'

print('Data folder:', DATA_DIR.resolve())
print('Files:', sorted(p.name for p in DATA_DIR.glob('*.csv')) if DATA_DIR.exists() else 'FOLDER NOT FOUND')
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec-load", r"""
## 0. Loading and repairing the data (technical)

`train.csv` is ~1.1 GB, so we read it **in chunks** (`chunksize`), as text. Missing values are encoded as `\N`.
Along the way we collect statistics about the raw files (row counts, missing values, how many "copies" of each
result exist) to show the scale of the problem described in section 3.

**The repair** (details in section 3): from each group of rows sharing a `resultId` we keep the one where
`raceId_y == racerId`. When no such row exists (~2% of results), we take the nearest earlier snapshot and
blank out the post-race standings columns — exactly as in notebook `02`.
"""),

cc("load", r"""
def _read_and_repair(path, chunksize=250_000):
    # One pass over the file: raw statistics + repair (1 row per resultId)
    exact_parts, fb_parts = [], []
    n_rows, null_cnt, per_id, columns = 0, None, None, None
    for chunk in pd.read_csv(path, dtype=str, na_values=NA_VALS, keep_default_na=False, chunksize=chunksize):
        chunk = chunk.replace('', np.nan)
        columns = list(chunk.columns)
        n_rows += len(chunk)
        null_cnt = chunk.isna().sum() if null_cnt is None else null_cnt.add(chunk.isna().sum(), fill_value=0)
        vc = chunk['resultId'].value_counts()
        per_id = vc if per_id is None else per_id.add(vc, fill_value=0)
        ry = pd.to_numeric(chunk['raceId_y'], errors='coerce')
        rc = pd.to_numeric(chunk['racerId'], errors='coerce')
        exact_parts.append(chunk[ry == rc])
        cand = chunk[ry < rc].assign(_ry=ry[ry < rc])
        if len(cand):
            fb_parts.append(cand.loc[cand.groupby('resultId')['_ry'].idxmax()])

    exact = pd.concat(exact_parts, ignore_index=True).drop_duplicates('resultId').assign(_recovered=0)
    missing_ids = set(per_id.index) - set(exact['resultId'])
    fb = (pd.concat(fb_parts, ignore_index=True) if fb_parts else exact.iloc[:0].assign(_ry=np.nan))
    fb = fb[fb['resultId'].isin(missing_ids)].sort_values('_ry').groupby('resultId').tail(1)
    fb = fb.drop(columns='_ry').assign(_recovered=1)
    fb.loc[:, [c for c in POST_RACE_STANDINGS if c in fb.columns]] = np.nan
    fixed = pd.concat([exact, fb], ignore_index=True)
    stats = dict(raw_rows=n_rows, n_cols=len(columns), columns=columns,
                 raw_null=(null_cnt / n_rows), per_id=per_id.astype(int),
                 n_results=len(per_id), n_exact=len(exact), n_recovered=len(fb),
                 n_dropped=len(missing_ids) - len(fb))
    return fixed, stats


def _cartesian_demo(path):
    # On the small file: which columns are constant within a result and which ones "jump"
    raw = pd.read_csv(path, dtype=str, na_values=NA_VALS, keep_default_na=False).replace('', np.nan)
    nun = raw.groupby('resultId').nunique(dropna=False).median().sort_values(ascending=False)
    rid = raw['resultId'].value_counts().index[len(raw['resultId'].unique()) // 2]   # a "typical" result
    cols = [c for c in ['resultId', 'racerId', 'driverId', 'grid', 'positionOrder',
                        'raceId_y', 'driverStandingsId', 'points_y', 'position', 'wins'] if c in raw.columns]
    example = raw.loc[raw['resultId'] == rid, cols].sort_values('raceId_y', key=lambda s: pd.to_numeric(s, errors='coerce'))
    return nun, example


if CACHE.exists():
    with open(CACHE, 'rb') as f:
        frames, raw_stats, demo_nunique, demo_example = pickle.load(f)
    print(f'Loaded from cache: {CACHE}  (delete the file to recompute from scratch)')
else:
    frames, raw_stats = {}, {}
    for split, fname in SPLITS.items():
        print(f'Reading {fname} ...', end=' ', flush=True)
        frames[split], raw_stats[split] = _read_and_repair(DATA_DIR / fname)
        s = raw_stats[split]
        print(f"{fmt_int(s['raw_rows'])} rows -> {fmt_int(len(frames[split]))} results")
    demo_nunique, demo_example = _cartesian_demo(DATA_DIR / SPLITS['validation'])
    with open(CACHE, 'wb') as f:
        pickle.dump((frames, raw_stats, demo_nunique, demo_example), f)
    print(f'Cache saved: {CACHE}')

for split, s in raw_stats.items():
    print(f"{split:>10}: {fmt_int(s['raw_rows']):>10} raw rows | {fmt_int(s['n_results']):>6} results "
          f"| exact snapshot {fmt_int(s['n_exact'])}, recovered {s['n_recovered']}, dropped {s['n_dropped']}")
"""),

cc("types", r"""
DATE_COLS = ['date', 'dob']
TEXT_COLS = {'time_x', 'time_y', 'fastestLapTime', 'positionText_x', 'positionText_y', 'status', 'grand_prix',
             'nationality', 'forename', 'surname', 'driverRef', 'driver_code', 'url_x', 'url_y'}

def convert_types(d):
    d = d.copy()
    for col in d.columns:
        if col in DATE_COLS:
            d[col] = pd.to_datetime(d[col], errors='coerce')
        elif col not in TEXT_COLS and not col.startswith('_') and d[col].notna().any():
            num = pd.to_numeric(d[col], errors='coerce')
            if num.notna().sum() >= 0.99 * d[col].notna().sum():
                d[col] = num
    return d

df = pd.concat([convert_types(f).assign(split=s) for s, f in frames.items()], ignore_index=True)
n_before = len(df)
df = df.drop_duplicates('resultId').sort_values(['year', 'round', 'positionOrder']).reset_index(drop=True)
if len(df) < n_before:
    print(f'Note: {n_before - len(df)} results appeared in two files — one copy kept.')

# ── Helper columns for the analysis ──────────────────────────────────────────
df['decade'] = (df['year'] // 10 * 10).astype(int)
df['class_A'] = pd.cut(df['positionOrder'], bins=[0, 1, 3, 10, np.inf], labels=CLASSES)

status_txt = df['status'].astype(str) if 'status' in df.columns else pd.Series('', index=df.index)
classified = (df['statusId'] == 1) | status_txt.str.startswith('+') | status_txt.eq('Finished')
df['dnf'] = (~classified).astype(int)

def status_group(s):
    s = str(s)
    if s == 'Finished' or s.startswith('+'):
        return 'Finished'
    if any(k in s for k in ['Accident', 'Collision', 'Spun', 'Crash', 'Damage', 'Debris', 'Fatal']):
        return 'Accident / collision'
    if any(k in s for k in ['qualify', 'Withdrew', 'Disqualified', 'Excluded', 'Not classified', '107%',
                            'Did not', 'Driver Seat', 'Injur', 'Illness', 'Eye']):
        return 'Did not start / DQ / other'
    if s in ('nan', 'None', ''):
        return 'No data'
    return 'Technical failure'
df['status_group'] = status_txt.map(status_group) if 'status' in df.columns else np.where(classified, 'Finished', 'Did not finish')

if {'forename', 'surname'} <= set(df.columns):
    df['driver_name'] = df['forename'].fillna('') + ' ' + df['surname'].fillna('')
elif 'driverRef' in df.columns:
    df['driver_name'] = df['driverRef'].str.replace('_', ' ').str.title()
else:
    df['driver_name'] = 'driver #' + df['driverId'].astype('Int64').astype(str)

# Team names: the dataset does NOT contain them — mapping of the main IDs from the public Ergast database
TEAM_NAMES = {1: 'McLaren', 2: 'BMW Sauber', 3: 'Williams', 4: 'Renault', 5: 'Toro Rosso', 6: 'Ferrari',
              7: 'Toyota', 9: 'Red Bull', 10: 'Force India', 11: 'Honda', 15: 'Sauber', 16: 'BAR',
              17: 'Jordan', 18: 'Minardi', 22: 'Benetton', 23: 'Brawn', 25: 'Tyrrell', 27: 'Ligier',
              32: 'Team Lotus', 34: 'Brabham', 51: 'Alfa Romeo', 117: 'Aston Martin', 131: 'Mercedes',
              210: 'Haas', 211: 'Racing Point', 213: 'AlphaTauri', 214: 'Alpine'}
team_col = next((c for c in ['constructor_name', 'constructorRef', 'name'] if c in df.columns), None)
df['team_name'] = (df[team_col] if team_col else
                   df['constructorId'].map(TEAM_NAMES).fillna('team #' + df['constructorId'].astype('Int64').astype(str)))

if {'date', 'dob'} <= set(df.columns):
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
df['places_gained'] = np.where(df['grid'] > 0, df['grid'] - df['positionOrder'], np.nan)

races = df.drop_duplicates('racerId')
print(f'Combined, repaired dataset: {fmt_int(len(df))} results × {df.shape[1]} columns (including helper columns)')
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec1", r"""
## 1. The dataset in numbers
"""),

cc("kpi", r"""
raw_total = sum(s['raw_rows'] for s in raw_stats.values())
n_cols_raw = max(s['n_cols'] for s in raw_stats.values())
kpis = [
    (fmt_int(raw_total), 'rows in the raw CSVs'),
    (fmt_int(len(df)), 'real results\n(driver × race)'),
    (f"{df['year'].nunique()}", f"seasons\n({int(df['year'].min())}–{int(df['year'].max())})"),
    (fmt_int(df['racerId'].nunique()), 'races (Grands Prix)'),
    (fmt_int(df['driverId'].nunique()), 'drivers'),
    (fmt_int(df['constructorId'].nunique()), 'teams (constructors)'),
    (fmt_int(df['circuitId'].nunique()), 'circuits'),
    (f'{n_cols_raw}', 'columns in the raw file'),
]
fig, axes = plt.subplots(2, 4, figsize=(14, 4.4))
for ax, (value, label) in zip(axes.flat, kpis):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle='round,pad=0,rounding_size=0.06',
                                transform=ax.transAxes, facecolor='white', edgecolor=GRID_C, linewidth=1))
    ax.text(0.5, 0.60, value, ha='center', va='center', fontsize=25, fontweight='bold', color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10, color=INK2, transform=ax.transAxes)
fig.suptitle('F1nalyze in numbers', x=0.01, ha='left', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
ratio = raw_total / len(df)
print(f'Note the first two numbers: there are ~{ratio:.0f}× more raw rows than real results — see section 3.')
"""),

mc("sec1-say", r"""
> 💬 **Talking point:** "The dataset covers practically the whole history of F1 — from the first race in 1950 up to 2023.
> It has over 3.5 million rows, but — as I will show in a moment — there are only about 26 thousand real observations."
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec2", r"""
## 2. Files and the time-based split

The organisers split the data **by time**, not randomly. This matters: the model learns from the past
and has to predict the future — just like in real life. So the files must not be mixed and rows must not be shuffled randomly.
"""),

cc("files-table", r"""
rows = []
for split, s in raw_stats.items():
    d = df[df['split'] == split]
    rows.append({'file': SPLITS[split], 'raw rows': fmt_int(s['raw_rows']), 'columns': s['n_cols'],
                 'results (resultId)': fmt_int(s['n_results']), 'inflation': f"×{s['raw_rows'] / s['n_results']:.0f}",
                 'years': f"{int(d['year'].min())}–{int(d['year'].max())}", 'races': d['racerId'].nunique(),
                 'has `position` label?': 'YES' if 'position' in s['columns'] else 'NO'})
ss = DATA_DIR / 'sample_submission.csv'
if ss.exists():
    sub = pd.read_csv(ss)
    rows.append({'file': ss.name, 'raw rows': fmt_int(len(sub)), 'columns': sub.shape[1],
                 'results (resultId)': '—', 'inflation': '—', 'years': '—', 'races': '—',
                 'has `position` label?': 'format: ' + ', '.join(sub.columns)})
display(pd.DataFrame(rows).set_index('file'))
"""),

cc("timeline", r"""
per_year = df.groupby(['year', 'split']).size().unstack(fill_value=0).reindex(columns=list(SPLITS), fill_value=0)
fig, ax = plt.subplots(figsize=(14, 4))
bottom = np.zeros(len(per_year))
for split in SPLITS:
    ax.bar(per_year.index, per_year[split], bottom=bottom, width=0.8, color=SPLIT_COLORS[split],
           label=f"{split} ({fmt_int(per_year[split].sum())} results)")
    bottom += per_year[split].values
for split in SPLITS:
    yrs = per_year.index[per_year[split] > 0]
    ax.annotate(split.upper(), xy=((yrs.min() + yrs.max()) / 2, bottom.max() * 1.04), ha='center',
                fontsize=10, fontweight='bold', color=SPLIT_COLORS[split])
ax.set_ylim(0, bottom.max() * 1.15)
ax.set_xlabel('Season'); ax.set_ylabel('Number of results (driver × race)')
ax.set_title('Time-based split: learn from the past, test on the future')
ax.legend(loc='upper left')
plt.tight_layout(); plt.show()
"""),

mc("sec2-say", r"""
> 💬 **Talking point:** "Train covers 1950–2013, validation 2013–2018, test 2018–2023. The number of results per season
> grows because the calendar grows — in the 1950s there were ~8 races a year, today there are more than 20."
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec3", r"""
## 3. ⚠️ Pitfall #1: the broken join (Cartesian product)

The raw files are **not** a "one row = one result" table. The organisers joined the results table with the
championship standings table (`driverStandings`) without the proper condition — every result had
**many standings snapshots from different rounds** glued onto it.

Consequence: the columns describing the race (`grid`, `positionOrder`, `driverId`…) are **constant** within a `resultId`,
but the standings columns (`position`, `points_y`, `driverStandingsId`) **take dozens of different values**.
The same feature row → many different labels. Without the repair the data is useless.
"""),

cc("cartesian-plot", r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1, 1.3]})
ax = axes[0]
names = list(raw_stats)
x = np.arange(len(names))
raw_rows = [raw_stats[s]['raw_rows'] for s in names]
real = [raw_stats[s]['n_results'] for s in names]
ax.bar(x - 0.2, raw_rows, 0.38, color=C['grey'], label='raw rows')
ax.bar(x + 0.2, real, 0.38, color=C['blue'], label='real results')
ax.set_yscale('log')
ax.set_xticks(x, names)
for i, (r, n) in enumerate(zip(raw_rows, real)):
    ax.text(i - 0.2, r * 1.15, fmt_int(r), ha='center', fontsize=8.5, color=INK2)
    ax.text(i + 0.2, n * 1.15, fmt_int(n), ha='center', fontsize=8.5, color=INK2)
    ax.text(i, r * 4, f'×{r / n:.0f}', ha='center', fontsize=13, fontweight='bold', color=INK)
ax.set_ylim(top=max(raw_rows) * 15)
ax.set_title('How many rows are "real"?')
ax.set_ylabel('number of rows (log scale)')
ax.legend(loc='upper right', ncol=2)

ax = axes[1]
per_id = raw_stats['train']['per_id']
step = max(1, int(np.ceil((per_id.max() - per_id.min() + 1) / 60)))
ax.hist(per_id.values, bins=np.arange(per_id.min() - 0.5, per_id.max() + step + 0.5, step),
        color=C['blue'], edgecolor=SURFACE, linewidth=0.5)
ax.axvline(per_id.median(), color=INK, ls='--', lw=1.2)
ax.text(per_id.median(), ax.get_ylim()[1] * 0.92, f'  median = {per_id.median():.0f} copies', fontsize=9.5)
ax.set_title('How many times is a single result repeated? (train)')
ax.set_xlabel('number of rows with the same resultId'); ax.set_ylabel('number of results')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout(); plt.show()
"""),

cc("cartesian-columns", r"""
nun = demo_nunique.drop(labels=['resultId'], errors='ignore')
nun = nun[nun > 1].sort_values()
fig, ax = plt.subplots(figsize=(10, max(2.5, 0.35 * len(nun) + 1)))
ax.barh(nun.index, nun.values, color=C['orange'], height=0.65)
for i, v in enumerate(nun.values):
    ax.text(v, i, f' {v:.0f}', va='center', fontsize=9, color=INK2)
ax.set_title('Columns that "jump" within a single result (validation)')
subtitle(ax, 'median number of distinct values per resultId — every other column has the value 1 (it is constant)')
ax.set_xlabel('median number of distinct values per resultId')
plt.tight_layout(); plt.show()
print('Columns constant within a result:', ', '.join(demo_nunique[demo_nunique <= 1].index))
"""),

mc("cartesian-example-md", r"""
**Example — one result (one driver in one race) in the raw file.** `grid` and `positionOrder` do not change,
but `position`, `points_y` and `driverStandingsId` come from different rounds. **Only** the row where
`raceId_y == racerId` is correct (highlighted).
"""),

cc("cartesian-example", r"""
ex = demo_example.head(12) if len(demo_example) <= 12 else pd.concat([
    demo_example[demo_example['raceId_y'] == demo_example['racerId']],
    demo_example.sample(11, random_state=1)]).drop_duplicates().sort_values(
        'raceId_y', key=lambda s: pd.to_numeric(s, errors='coerce'))
print(f'This result has {len(demo_example)} copies in the file; showing {len(ex)}.')
try:
    display(ex.style.apply(lambda r: ['background-color: #fdf0c8; font-weight: bold'
                                      if r['raceId_y'] == r['racerId'] else '' for _ in r], axis=1)
              .hide(axis='index'))
except Exception:
    display(ex)
"""),

mc("cartesian-fix", r"""
### The fix
1. Keep the row with `raceId_y == racerId` → **exactly 1 row per `resultId`**.
2. For ~2% of results there is no such row → take the **nearest earlier** snapshot and blank the post-race columns
   (`points_y`, `wins`, `position`, `positionText_y`). Nothing from the future leaks into the features.
3. Mind the names: **`racerId` is a race ID, not a driver ID!** The driver is identified by `driverId`.
"""),

cc("repair-summary", r"""
rep = pd.DataFrame({s: {'results in file': st['n_results'], 'exact snapshot': st['n_exact'],
                        'recovered (earlier snapshot)': st['n_recovered'], 'dropped': st['n_dropped']}
                    for s, st in raw_stats.items()}).T
rep['% kept'] = ((rep['exact snapshot'] + rep['recovered (earlier snapshot)'])
                 / rep['results in file'] * 100).round(2)
display(rep)
"""),

mc("sec3-say", r"""
> 💬 **Talking point:** "This is the most important finding of the exploration phase. The raw file suggested 2.8 million
> observations and 108 classes with an imbalance of around 170,000×. In reality that was an artefact of a bad join."
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec4", r"""
## 4. Variable dictionary

We describe every column with four questions: **what does it mean**, **which group does it belong to**,
**when is it known** (before the start or only after the race) and **what role can it play** in a model.

The "when known" column is crucial: anything that is known **only after the race** is *data leakage* —
a model that uses it "peeks at the answer".
"""),

cc("dictionary", r"""
# (group, description, when known, role)
VAR_INFO = {
    'resultId':              ('Keys / IDs', 'result ID (driver × race) — the observation key', 'meta', 'ID / key'),
    'racerId':               ('Keys / IDs', 'RACE ID (despite the name!) — which race the result belongs to', 'meta', 'ID / key'),
    'driverId':              ('Keys / IDs', 'driver ID', 'before', 'ID / key'),
    'constructorId':         ('Keys / IDs', 'team (constructor) ID', 'before', 'feature'),
    'circuitId':             ('Race', 'circuit ID', 'before', 'feature'),
    'statusId':              ('Race result', 'finishing status ID (1 = Finished)', 'after', 'leakage'),
    'result_driver_standing':('Keys / IDs', 'row ID in the competition submission', 'meta', 'ID / key'),
    'number':                ('Driver', 'car number in this race', 'before', 'drop'),
    'grid':                  ('Race', 'starting position (1 = pole position, 0 = pit-lane start)', 'before', 'feature'),
    'year':                  ('Race', 'season', 'before', 'feature'),
    'round':                 ('Race', 'round number within the season', 'before', 'feature'),
    'grand_prix':            ('Race', 'Grand Prix name', 'before', 'feature'),
    'date':                  ('Race', 'race date', 'before', 'feature'),
    'time_y':                ('Race', 'race start time', 'before', 'drop'),
    'url_x':                 ('Metadata', 'Wikipedia link', 'meta', 'drop'),
    'url_y':                 ('Metadata', 'Wikipedia link', 'meta', 'drop'),
    'position_x':            ('Race result', 'finishing position (empty if not classified)', 'after', 'leakage'),
    'positionText_x':        ('Race result', 'position as text (R = retired, D = DQ…)', 'after', 'leakage'),
    'positionOrder':         ('Race result', 'final order (always filled) — TARGET VARIABLE', 'after', 'TARGET'),
    'points':                ('Race result', 'points scored in this race', 'after', 'leakage'),
    'laps':                  ('Race result', 'number of laps completed', 'after', 'leakage'),
    'time_x':                ('Race result', 'race time / gap to the winner (text)', 'after', 'leakage'),
    'timetaken_in_millisec': ('Race result', 'race time in milliseconds', 'after', 'leakage'),
    'fastestLap':            ('Race result', 'lap number of the fastest lap', 'after', 'leakage'),
    'rank':                  ('Race result', 'fastest-lap ranking', 'after', 'leakage'),
    'fastestLapTime':        ('Race result', 'fastest lap time', 'after', 'leakage'),
    'max_speed':             ('Race result', 'speed on the fastest lap [km/h]', 'after', 'leakage'),
    'status':                ('Race result', 'status: Finished / +1 Lap / Engine / Accident…', 'after', 'leakage'),
    'driverStandingsId':     ('Championship standings', 'ID of the drivers\' standings snapshot', 'meta', 'ID / key'),
    'raceId_y':              ('Championship standings', 'race after which the snapshot was taken (repair key)', 'meta', 'ID / key'),
    'points_y':              ('Championship standings', 'season points total AFTER this race', 'after', 'lagged feature'),
    'position':              ('Championship standings', 'standings position AFTER the race (competition target; missing in test)', 'after', 'leakage'),
    'positionText_y':        ('Championship standings', 'standings position as text', 'after', 'leakage'),
    'wins':                  ('Championship standings', 'number of season wins AFTER this race', 'after', 'lagged feature'),
    'driverRef':             ('Driver', 'driver text identifier', 'before', 'drop'),
    'driver_num':            ('Driver', 'permanent driver number (since 2014)', 'before', 'drop'),
    'driver_code':           ('Driver', 'three-letter code (e.g. HAM, VER)', 'before', 'drop'),
    'forename':              ('Driver', 'first name', 'before', 'drop'),
    'surname':               ('Driver', 'last name', 'before', 'drop'),
    'dob':                   ('Driver', 'date of birth → age on race day', 'before', 'feature'),
    'nationality':           ('Driver', 'driver nationality', 'before', 'feature'),
}
for s in ['fp1', 'fp2', 'fp3', 'quali', 'sprint']:
    for t in ['date', 'time']:
        VAR_INFO[f'{s}_{t}'] = ('Sessions (empty)', f'{s.upper()} session {t} — always empty in this dataset', 'before', 'drop')

all_cols = list(dict.fromkeys(c for s in raw_stats.values() for c in s['columns']))
raw_null_all = pd.concat([s['raw_null'] for s in raw_stats.values()], axis=1).mean(axis=1)
rows = []
for col in all_cols:
    grp, desc, when, role = VAR_INFO.get(col, ('Other', '(column not in the dictionary)', '?', '?'))
    s = df[col] if col in df.columns else pd.Series(dtype=object)
    ex = s.dropna()
    rows.append({'column': col, 'group': grp, 'description': desc, 'when known': when, 'role': role,
                 'type': ('date' if pd.api.types.is_datetime64_any_dtype(s) else
                          'number' if pd.api.types.is_numeric_dtype(s) else 'text'),
                 '% missing': round(s.isna().mean() * 100, 1) if len(s) else 100.0,
                 'unique': s.nunique(),
                 'in files': '/'.join(k[:5] for k, st in raw_stats.items() if col in st['columns']),
                 'example': str(ex.iloc[len(ex) // 2])[:22] if len(ex) else '—'})
var_dict = pd.DataFrame(rows)
ROLE_ORDER = ['TARGET', 'feature', 'lagged feature', 'leakage', 'ID / key', 'drop', '?']
var_dict['_o'] = var_dict['role'].map({r: i for i, r in enumerate(ROLE_ORDER)})
var_dict = var_dict.sort_values(['group', '_o', 'column']).drop(columns='_o').reset_index(drop=True)

ROLE_COLORS = {'TARGET': '#fdf0c8', 'feature': '#d9f2e7', 'lagged feature': '#e6f0fb',
               'leakage': '#fde3da', 'ID / key': '#f0efec', 'drop': '#f7f7f5', '?': 'white'}
try:
    display(var_dict.style
            .apply(lambda r: [f"background-color: {ROLE_COLORS.get(r['role'], 'white')}"] * len(r), axis=1)
            .bar(subset=['% missing'], color='#9ec5f4', vmin=0, vmax=100)
            .format({'% missing': '{:.1f}'}).hide(axis='index'))
except Exception:
    display(var_dict)
print('Colour legend: yellow = target, green = feature, blue = lagged feature, '
      'red = leakage (known after the race), grey = ID / drop')
"""),

cc("dictionary-plot", r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
grp = var_dict['group'].value_counts().sort_values()
axes[0].barh(grp.index, grp.values, color=C['blue'], height=0.65)
for i, v in enumerate(grp.values):
    axes[0].text(v, i, f' {v}', va='center', fontsize=9.5, color=INK2)
axes[0].set_title('Columns by topic group')
axes[0].set_xlabel('number of columns')

role_col = {'TARGET': C['yellow'], 'feature': C['aqua'], 'lagged feature': C['blue'],
            'leakage': C['orange'], 'ID / key': '#898781', 'drop': C['grey'], '?': '#e1e0d9'}
role = var_dict['role'].value_counts().reindex(ROLE_ORDER).dropna()[::-1]
axes[1].barh(role.index, role.values, color=[role_col[r] for r in role.index], height=0.65)
for i, v in enumerate(role.values):
    axes[1].text(v, i, f' {int(v)}', va='center', fontsize=9.5, color=INK2)
axes[1].set_title('What is each column good for?')
axes[1].set_xlabel('number of columns')
plt.tight_layout(); plt.show()
n_useful = int(var_dict['role'].isin(['feature', 'lagged feature']).sum())
print(f'Of {len(var_dict)} columns only {n_useful} can be used directly as features (the rest are IDs, leakage or junk).')
"""),

mc("one-row-md", r"""
### What does one (repaired) row look like?
The last win in the dataset — every column with its description:
"""),

cc("one-row", r"""
winner = df[df['positionOrder'] == 1].sort_values(['year', 'round']).iloc[-1]
row = (pd.DataFrame({'value': winner.reindex(all_cols).astype(str).str[:40]})
       .join(var_dict.set_index('column')[['description', 'role']]))
display(row)
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec5", r"""
## 5. Missing values — when and why

Missing values in this dataset are **not random**: they depend on the era (e.g. lap times are recorded only from 2004)
and on whether the driver reached the finish (no race time = did not finish).
"""),

cc("missing-bar", r"""
miss = pd.DataFrame({
    'raw files': raw_null_all,
    'after repair': pd.Series({c: df[c].isna().mean() for c in all_cols if c in df.columns}),
}).fillna(1.0)
miss = miss[miss.max(axis=1) > 0].sort_values('after repair')
fig, ax = plt.subplots(figsize=(11, max(3, 0.3 * len(miss) + 1)))
y = np.arange(len(miss))
ax.barh(y + 0.2, miss['raw files'], 0.38, color=C['grey'], label='raw files')
ax.barh(y - 0.2, miss['after repair'], 0.38, color=C['blue'], label='after repair')
ax.set_yticks(y, miss.index)
pct_axis(ax, 'x')
ax.set_title('Share of missing values per column (all files)')
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()
always_empty = miss.index[miss['after repair'] >= 0.999].tolist()
print(f'Columns that are 100% empty ({len(always_empty)}): {", ".join(always_empty)}')
"""),

cc("missing-heatmap", r"""
cols_m = [c for c in all_cols if c in df.columns and 0 < df[c].isna().mean() < 0.999]
heat = df.groupby('decade')[cols_m].apply(lambda g: g.isna().mean()).T
heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(11, max(3, 0.38 * len(heat) + 1)))
sns.heatmap(heat, cmap=SEQ_BLUE, vmin=0, vmax=1, annot=True, fmt='.0%', annot_kws={'fontsize': 8},
            linewidths=1.5, linecolor=SURFACE, cbar_kws={'format': mticker.PercentFormatter(1.0), 'shrink': 0.7}, ax=ax)
ax.set_xticklabels([f'{int(float(t.get_text()))}s' for t in ax.get_xticklabels()], rotation=0)
ax.set_xlabel('decade'); ax.set_ylabel('')
ax.set_title('Missingness map: column × decade')
subtitle(ax, 'darker = more missing; shows the eras in which the data simply was not collected')
plt.tight_layout(); plt.show()
"""),

mc("sec5-say", r"""
> 💬 **Talking point:** "The 10 columns about practice sessions and qualifying are completely empty — qualifying data
> would be very valuable, but it is not here. Lap times and speeds appear only in the 21st century,
> and only drivers who reached the finish have a race time."
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec6", r"""
## 6. The target variable: `positionOrder` → 4 classes

`positionOrder` is the final finishing order — it is filled for **every** driver, including those who did not finish.
We group it into 4 classes with a sporting meaning: **win (1)**, **podium (2–3)**, **points (4–10)**, **rest (>10)**.

*Why not `position` (the Kaggle competition target)?* It is the position in the **championship standings** after the race,
not the race result. It is strongly autocorrelated with the previous round and is missing from the test file.
"""),

cc("target", r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={'width_ratios': [1.4, 1]})
ax = axes[0]
po = df['positionOrder'].value_counts().sort_index()
po = po[po.index <= 30]
cls_of = pd.cut(po.index, bins=[0, 1, 3, 10, np.inf], labels=CLASSES)
ax.bar(po.index, po.values, color=[CLASS_COLORS[c] for c in cls_of], width=0.82)
ax.set_title('Distribution of positionOrder (all years)')
ax.set_xlabel('final position'); ax.set_ylabel('number of results')
ax.xaxis.set_major_locator(mticker.MultipleLocator(1 if len(po) <= 20 else 2))
ax.grid(axis='x', visible=False)
ax.legend(handles=[Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in CLASSES],
          loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4)

ax = axes[1]
share = df.groupby('split')['class_A'].value_counts(normalize=True).unstack().reindex(index=list(SPLITS)[::-1], columns=CLASSES)
left = np.zeros(len(share))
for c in CLASSES:
    ax.barh(share.index, share[c], left=left, color=CLASS_COLORS[c], height=0.6, edgecolor=SURFACE, linewidth=2)
    for i, (l, v) in enumerate(zip(left, share[c])):
        if v > 0.06:
            ax.text(l + v / 2, i, f'{v:.0%}', ha='center', va='center', fontsize=9.5,
                    color=INK if c in ('win', 'rest') else 'white', fontweight='bold')
    left += share[c].values
pct_axis(ax, 'x'); ax.grid(False)
ax.set_title('Class shares in each file')
plt.tight_layout(); plt.show()

cnt = df['class_A'].value_counts().reindex(CLASSES)
print('Class counts (whole dataset):', {CLASS_LABELS[k]: int(v) for k, v in cnt.items()})
print(f'Imbalance: the "rest" class is {cnt.max() / cnt.min():.0f}× larger than "win".')
print('Takeaway: accuracy would be misleading → we use macro-F1 and balanced accuracy.')
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec7", r"""
## 7. Exploration — what does the data say about Formula 1?

### 7.1 How F1 changed over 74 seasons
"""),

cc("seasons", r"""
season = pd.DataFrame({
    'races per season': df.groupby('year')['racerId'].nunique(),
    'average starters per race': df.groupby(['year', 'racerId']).size().groupby('year').mean(),
    'distinct drivers per season': df.groupby('year')['driverId'].nunique(),
})
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharex=True)
for ax, (col, color) in zip(axes, zip(season.columns, [C['blue'], C['orange'], C['aqua']])):
    ax.plot(season.index, season[col], color=color)
    ax.fill_between(season.index, season[col], alpha=0.12, color=color, linewidth=0)
    ax.set_title(col.capitalize())
    ax.set_ylim(bottom=0)
    last = season[col].dropna()
    ax.annotate(f'{last.iloc[-1]:.0f}', (last.index[-1], last.iloc[-1]), xytext=(4, 0),
                textcoords='offset points', fontsize=9, color=INK2, va='center')
fig_title(fig, 'How the scale of F1 changed over 74 seasons')
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.show()
"""),

mc("sec7-2", r"""
### 7.2 The starting grid — the strongest signal in the dataset
"""),

cc("grid-heatmap", r"""
g = df[(df['grid'].between(1, 20)) & (df['positionOrder'] <= 20)]
ct = pd.crosstab(g['grid'], g['positionOrder'], normalize='index')
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(ct, cmap=SEQ_BLUE, vmin=0, vmax=ct.values.max(), linewidths=0.5, linecolor=SURFACE,
            cbar_kws={'format': mticker.PercentFormatter(1.0, decimals=0), 'shrink': 0.7, 'label': 'P(finish | grid)'}, ax=ax)
ax.plot([0, 20], [0, 20], color=C['orange'], lw=1.2, ls='--')
ax.set_xlabel('finishing position (positionOrder)'); ax.set_ylabel('starting position (grid)')
ax.set_title('Where you start is (often) where you finish')
subtitle(ax, 'each row sums to 100%; orange diagonal = "finish = start"')
plt.tight_layout(); plt.show()
print(f"Spearman correlation grid ↔ positionOrder (grid>0): {df.loc[df['grid'] > 0, ['grid', 'positionOrder']].corr('spearman').iloc[0, 1]:.2f}")
"""),

cc("grid-class", r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axes[0]
gk = df[df['grid'].between(1, 24)].groupby('grid')['class_A'].value_counts(normalize=True).unstack().reindex(columns=CLASSES).fillna(0)
bottom = np.zeros(len(gk))
for c in CLASSES:
    ax.bar(gk.index, gk[c], bottom=bottom, color=CLASS_COLORS[c], width=0.8, label=CLASS_LABELS[c],
           edgecolor=SURFACE, linewidth=1)
    bottom += gk[c].values
pct_axis(ax); ax.grid(axis='x', visible=False)
ax.set_xticks(gk.index)
ax.set_xlabel('starting position (grid)')
ax.set_title('Result class by starting position')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))

ax = axes[1]
pole = df[df['grid'] == 1].groupby('decade')['positionOrder'].apply(lambda s: (s == 1).mean())
ax.bar([f'{d}s' for d in pole.index], pole.values, color=C['yellow'], width=0.65)
for i, v in enumerate(pole.values):
    ax.text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=9, color=INK2)
pct_axis(ax); ax.set_ylim(0, max(pole.max() * 1.2, 0.6))
ax.set_title('How often does pole position = victory?')
plt.tight_layout(); plt.show()
"""),

cc("places-gained", r"""
fin = df[(df['dnf'] == 0) & (df['grid'] > 0)]
fig, ax = plt.subplots(figsize=(12, 3.8))
vals = fin['places_gained'].clip(-15, 15)
bins = np.arange(-15.5, 16.5, 1)
n, b, patches = ax.hist(vals, bins=bins, color=C['grey'], edgecolor=SURFACE)
for p, left in zip(patches, b[:-1]):
    p.set_facecolor(C['aqua'] if left + 0.5 > 0 else (C['orange'] if left + 0.5 < 0 else C['grey']))
ax.set_xlabel('places gained (grid − positionOrder); > 0 = gained, < 0 = lost; clipped to ±15')
ax.set_ylabel('number of results')
ax.set_title('How many places do drivers gain during the race? (finishers only)')
plt.tight_layout(); plt.show()
print(f"Median: {fin['places_gained'].median():+.0f} places | gained: {(fin['places_gained'] > 0).mean():.0%} "
      f"| unchanged: {(fin['places_gained'] == 0).mean():.0%} | lost: {(fin['places_gained'] < 0).mean():.0%}")
print('(Drivers who finish usually gain places — because someone ahead of them retired.)')
"""),

mc("sec7-2-say", r"""
> 💬 **Talking point:** "This is the key chart of the project: our class boundaries (1 / 2–3 / 4–10 / >10) line up
> with bands of the starting grid. That is why the simple rule 'class = starting position' turned out to be a very strong baseline."
"""),

mc("sec7-3", r"""
### 7.3 Failures and accidents (DNF — *Did Not Finish*)
"""),

cc("status", r"""
SG_COLORS = {'Finished': C['aqua'], 'Technical failure': C['orange'], 'Accident / collision': C['red'],
             'Did not start / DQ / other': C['violet'], 'No data': C['grey'], 'Did not finish': C['orange']}
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
if 'status' in df.columns:
    top = df['status'].value_counts().head(15)[::-1]
    ax.barh(top.index, top.values, color=[SG_COLORS[status_group(s)] for s in top.index], height=0.7)
    for i, v in enumerate(top.values):
        ax.text(v, i, f' {fmt_int(v)}', va='center', fontsize=8.5, color=INK2)
    ax.set_title('The 15 most common statuses')
    groups_present = [k for k in SG_COLORS if k in df['status_group'].unique()]
    ax.legend(handles=[Patch(color=SG_COLORS[k], label=k) for k in groups_present], loc='lower right')

ax = axes[1]
dec = df.groupby('decade')['status_group'].value_counts(normalize=True).unstack().fillna(0)
order = [k for k in SG_COLORS if k in dec.columns]
bottom = np.zeros(len(dec))
xl = [f'{d}s' for d in dec.index]
for k in order:
    ax.bar(xl, dec[k], bottom=bottom, color=SG_COLORS[k], label=k, width=0.7, edgecolor=SURFACE, linewidth=1.5)
    bottom += dec[k].values
pct_axis(ax); ax.grid(axis='x', visible=False)
ax.set_title('How drivers ended the race — by decade')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)
plt.tight_layout(); plt.show()
"""),

cc("dnf-year", r"""
dnf_year = df.groupby('year')['dnf'].mean()
fig, ax = plt.subplots(figsize=(14, 3.8))
ax.plot(dnf_year.index, dnf_year.values, color=C['red'], lw=1.2, alpha=0.45, marker='o', ms=3)
roll = dnf_year.rolling(5, center=True, min_periods=2).mean()
ax.plot(roll.index, roll.values, color=C['red'], lw=2.5, label='5-year rolling mean')
pct_axis(ax); ax.set_ylim(0, 1)
for yr in [dnf_year.idxmax(), dnf_year.idxmin()]:
    ax.annotate(f'{yr}: {dnf_year[yr]:.0%}', (yr, dnf_year[yr]), xytext=(6, 6 if yr == dnf_year.idxmax() else -14),
                textcoords='offset points', fontsize=9, color=INK2)
ax.set_title('Share of drivers who did NOT reach the finish (DNF), per season')
ax.legend(loc='upper right')
plt.tight_layout(); plt.show()
print(f"Mean DNF rate: {df['dnf'].mean():.1%}  (note: also counts non-qualifiers in the early years)")
"""),

mc("sec7-3-say", r"""
> 💬 **Talking point:** "Cars have become far more reliable — in the past half of the field did not reach the finish,
> today it is in the low teens of percent. This is an example of **drift over time**: the same feature means something different in different eras."
"""),

mc("sec7-4", r"""
### 7.4 Legends: drivers, teams, nationalities
"""),

cc("legends", r"""
wins = df[df['positionOrder'] == 1]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
top_d = wins['driver_name'].value_counts().head(15)[::-1]
axes[0].barh(top_d.index, top_d.values, color=C['yellow'], height=0.7)
for i, v in enumerate(top_d.values):
    axes[0].text(v, i, f' {v}', va='center', fontsize=9, color=INK2)
axes[0].set_title('Most wins — drivers')
axes[0].set_xlabel('number of race wins')

top_t = wins['team_name'].value_counts().head(12)[::-1]
axes[1].barh(top_t.index, top_t.values, color=C['blue'], height=0.7)
for i, v in enumerate(top_t.values):
    axes[1].text(v, i, f' {v}', va='center', fontsize=9, color=INK2)
axes[1].set_title('Most wins — teams')
axes[1].set_xlabel('number of race wins')
plt.tight_layout(); plt.show()
if not team_col:
    print('Note: the dataset has no team names — names were added by hand from the IDs in the public Ergast database.')
"""),

cc("nationality-age", r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1, 1.4]})
nat = df.drop_duplicates('driverId')['nationality'].value_counts().head(12)[::-1]
axes[0].barh(nat.index, nat.values, color=C['aqua'], height=0.7)
for i, v in enumerate(nat.values):
    axes[0].text(v, i, f' {v}', va='center', fontsize=9, color=INK2)
axes[0].set_title('Driver nationalities (unique drivers)')
pl = df.drop_duplicates('driverId')['nationality'].eq('Polish').sum()
if pl:
    axes[0].text(0.98, 0.04, f'Polish drivers in the dataset: {pl}', transform=axes[0].transAxes, ha='right', fontsize=9, color=INK2)

if 'driver_age' in df.columns:
    ax = axes[1]
    d = df[df['driver_age'].between(15, 60)]
    decs = sorted(d['decade'].unique())
    ax.boxplot([d.loc[d['decade'] == x, 'driver_age'] for x in decs], tick_labels=[f'{x}s' for x in decs],
               widths=0.55, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='#cde2fb', edgecolor=C['blue']), medianprops=dict(color=INK, lw=2),
               whiskerprops=dict(color=C['blue']), capprops=dict(color=C['blue']))
    w = d[d['positionOrder'] == 1].groupby('decade')['driver_age'].median()
    ax.plot([decs.index(x) + 1 for x in w.index], w.values, 'o', color=C['yellow'], ms=9,
            markeredgecolor=INK, label='median age of winners')
    ax.set_title('Driver age on race day — by decade')
    ax.set_ylabel('age [years]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(); plt.show()
"""),

mc("sec7-5", r"""
### 7.5 The points system changed — points from different eras are not comparable
"""),

cc("points-system", r"""
pts = df[df['positionOrder'] == 1].groupby('year')['points'].median()
top10_scoring = df[df['points'] > 0].groupby('year')['positionOrder'].max()
fig, axes = plt.subplots(1, 2, figsize=(15, 3.9))
axes[0].step(pts.index, pts.values, where='mid', color=C['violet'])
for yr in pts.index[pts.diff().fillna(0).abs() >= 1]:
    axes[0].annotate(f'{pts[yr]:g} pts ({yr})', (yr, pts[yr]), xytext=(-8, 7), textcoords='offset points', fontsize=8.5, color=INK2)
axes[0].set_title('Points for a win (season median)')
axes[0].set_ylim(0, pts.max() * 1.25)
axes[1].step(top10_scoring.index, top10_scoring.values, where='mid', color=C['aqua'])
axes[1].set_title('Down to which position were points awarded?')
axes[1].set_ylim(0, top10_scoring.max() + 3)
plt.tight_layout(); plt.show()
print('Takeaway: before 2003 only the top 5/6/8 scored → in the old data the "points (4–10)" class is a band of positions, not actual points.')
"""),

mc("sec7-6", r"""
### 7.6 Circuits and speed
"""),

cc("circuits", r"""
gp_col = 'grand_prix' if 'grand_prix' in df.columns else 'circuitId'
gps = races[gp_col].value_counts().head(15)[::-1]
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(gps.index.astype(str), gps.values, color=C['blue'], height=0.7)
for i, v in enumerate(gps.values):
    ax.text(v, i, f' {v}', va='center', fontsize=9, color=INK2)
ax.set_title('Most frequently held Grands Prix')
ax.set_xlabel('number of races in the dataset')
plt.tight_layout(); plt.show()
"""),

cc("speed", r"""
if 'max_speed' in df.columns:
    sp = df.groupby('year')['max_speed'].agg(['median', lambda s: s.notna().mean()])
    sp.columns = ['median', 'coverage']
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.8), sharex=True)
    axes[0].bar(sp.index, sp['coverage'], color=C['grey'], width=0.8)
    pct_axis(axes[0]); axes[0].set_title('For how many results do we know the speed?')
    s = sp['median'].dropna()
    axes[1].plot(s.index, s.values, color=C['orange'], marker='o', ms=4)
    axes[1].set_title('Median fastest-lap speed [km/h]')
    plt.tight_layout(); plt.show()
    print(f"Speed available from {int(s.index.min()) if len(s) else '—'} — before that the column is empty.")
"""),

cc("correlation", r"""
corr_cols = [c for c in ['grid', 'positionOrder', 'points', 'laps', 'year', 'round', 'driver_age',
                         'points_y', 'wins', 'position', 'max_speed', 'dnf'] if c in df.columns]
corr = df[corr_cols].corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
            annot_kws={'fontsize': 8.5}, linewidths=1.5, linecolor=SURFACE, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Spearman correlation'}, ax=ax)
ax.set_title('Correlations between numeric variables')
subtitle(ax, 'note: points, laps, dnf, position are POST-race variables — for description only, not as features')
plt.tight_layout(); plt.show()
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec8", r"""
## 8. Fun facts and records from the data 🏆
"""),

cc("records", r"""
starts = df.groupby('driver_name').size()
wins_d = df[df['positionOrder'] == 1].groupby('driver_name').size()
pods_d = df[df['positionOrder'] <= 3].groupby('driver_name').size()
w = df[df['positionOrder'] == 1]
facts = {
    'Most starts': f'{starts.idxmax()} — {starts.max()}',
    'Most wins': f'{wins_d.idxmax()} — {wins_d.max()}',
    'Most podiums': f'{pods_d.idxmax()} — {pods_d.max()}',
    'Best win rate (min. 50 starts)': (lambda r: f'{r.idxmax()} — {r.max():.0%} of races won')(
        (wins_d.reindex(starts.index).fillna(0) / starts)[starts >= 50]),
}
if 'driver_age' in df.columns:
    young, old = w.loc[w['driver_age'].idxmin()], w.loc[w['driver_age'].idxmax()]
    facts['Youngest winner'] = f"{young['driver_name']} — {young['driver_age']:.1f} years ({int(young['year'])})"
    facts['Oldest winner'] = f"{old['driver_name']} — {old['driver_age']:.1f} years ({int(old['year'])})"
far = w[w['grid'] > 0].sort_values('grid').iloc[-1]
facts['Win from the furthest grid slot'] = f"{far['driver_name']} — started P{int(far['grid'])}, {far.get(gp_col, '')} {int(far['year'])}"
facts['Most unpredictable season (most distinct winners)'] = (
    lambda s: f'{s.idxmax()} — {s.max()} different winners')(w.groupby('year')['driverId'].nunique())
display(pd.DataFrame.from_dict(facts, orient='index', columns=['']).rename_axis('record'))
"""),

cc("interactive", r"""
# Bonus: interactive charts (hover with the mouse). Requires: pip install plotly
try:
    import plotly.express as px
    car = (df.groupby(['driver_name', 'nationality'])
             .agg(starts=('resultId', 'size'), wins=('positionOrder', lambda s: (s == 1).sum()),
                  podiums=('positionOrder', lambda s: (s <= 3).sum()), first=('year', 'min'), last=('year', 'max'))
             .reset_index())
    tm = car[car['wins'] > 0]
    fig = px.treemap(tm, path=[px.Constant('All winners'), 'nationality', 'driver_name'], values='wins',
                     color='wins', color_continuous_scale=['#cde2fb', '#3987e5', '#0d366b'],
                     title='Wins: nationality → driver (click to zoom in)')
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=550)
    fig.show()

    fig = px.scatter(car[car['starts'] >= 20], x='starts', y='wins', size='podiums', size_max=40,
                     hover_name='driver_name', hover_data=['nationality', 'first', 'last'],
                     color_discrete_sequence=[C['blue']], opacity=0.6,
                     title='Driver careers: starts vs wins (size = podiums, min. 20 starts)')
    fig.update_layout(height=500, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE)
    fig.show()
except ImportError:
    print('plotly is not installed — skipping the interactive charts (pip install plotly).')
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec9", r"""
## 9. Drift over time — the past ≠ the future

A model trained on 1950–2013 has to work on 2018–2023. How much of what it knows even exists in the future?
"""),

cc("coverage", r"""
train_ids = {k: set(df.loc[df['split'] == 'train', k].dropna()) for k in ['driverId', 'constructorId', 'circuitId']}
labels = {'driverId': 'driver', 'constructorId': 'team', 'circuitId': 'circuit'}
cov = pd.DataFrame({s: {labels[k]: (~df.loc[df['split'] == s, k].isin(v)).mean() for k, v in train_ids.items()}
                    for s in ['validation', 'test']})
fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(cov))
for i, s in enumerate(cov.columns):
    ax.bar(x + (i - 0.5) * 0.38, cov[s], 0.36, color=SPLIT_COLORS[s], label=s)
    for xi, v in zip(x, cov[s]):
        ax.text(xi + (i - 0.5) * 0.38, v + 0.015, f'{v:.0%}', ha='center', fontsize=9.5, color=INK2)
ax.set_xticks(x, cov.index)
pct_axis(ax); ax.set_ylim(0, max(0.3, cov.values.max() * 1.25))
ax.set_title('What % of rows involve a driver / team / circuit that is NOT in train?')
ax.legend(loc='upper right')
plt.tight_layout(); plt.show()
print('Takeaway: in test the raw driverId is mostly "strangers" → instead of IDs we need historical features (form, experience).')
"""),

cc("drift-summary", r"""
drift = df.groupby('split').agg(
    seasons=('year', lambda s: f'{int(s.min())}–{int(s.max())}'),
    races_per_season=('racerId', lambda s: s.nunique()),
    starters_per_race=('racerId', lambda s: s.value_counts().mean()),
    DNF=('dnf', 'mean'),
    driver_age=('driver_age', 'median') if 'driver_age' in df.columns else ('year', 'size'),
    pole_wins=('grid', lambda s: ((s == 1) & (df.loc[s.index, 'positionOrder'] == 1)).sum() / max((s == 1).sum(), 1)),
).reindex(list(SPLITS))
drift['races_per_season'] = (drift['races_per_season'] / df.groupby('split')['year'].nunique()).round(1)
drift['starters_per_race'] = drift['starters_per_race'].round(1)
drift['DNF'] = (drift['DNF'] * 100).round(1).astype(str) + '%'
drift['pole_wins'] = (drift['pole_wins'] * 100).round(1).astype(str) + '%'
if 'driver_age' in df.columns:
    drift['driver_age'] = drift['driver_age'].round(1)
display(drift)
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec10", r"""
## 10. ✅ Strengths and ❌ weaknesses of the dataset

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| **Huge time span** — 74 seasons, practically the whole history of F1 | **Broken join** — rows artificially duplicated ~128–193×; without the repair the data is useless |
| **Complete key columns** — `grid`, `positionOrder`, `year`, `round`, IDs have no missing values | **10 columns 100% empty** — no qualifying or practice data, i.e. the most valuable pre-race information |
| **Rich description of the result** — status, laps, times, points, speed | **Most columns are leakage** — known only after the race; they cannot be features |
| **Ready-made time-based split** train / val / test — realistic "forward" evaluation | **Misleading names** — `racerId` is a race ID, `position` is the championship standing, `_x`/`_y` suffixes from the join |
| **Natural, understandable target** with a clear sporting interpretation | **Strong drift** — different rules, points systems, reliability and field sizes in every era |
| **Relational structure (Ergast)** — enables historical features: form, experience, circuit statistics | **Non-random missingness (MNAR)** — times/speed only for recent years and only for finishers |
| **Interesting domain** — results are easy to interpret and sanity-check with F1 knowledge | **No team names, weather, strategy or pit stops** — important factors are absent |
| | **Imbalanced classes** (~14×) and little data: only ~26k real observations |
| | **Inconsistent competition target** — `sample_submission` expects `position`, which is missing from `test.csv` |
"""),

cc("scorecard", r"""
# Subjective rating of the dataset (1–5) — for a single slide
score = pd.Series({
    'Time span': 5, 'Completeness of key columns': 4, 'Interpretability': 4,
    'ML-readiness (without cleaning)': 1, 'Richness of pre-race features': 2,
    'Consistency over time (no drift)': 2, 'Documentation / naming': 2,
}).sort_values()
fig, ax = plt.subplots(figsize=(10, 3.9))
cols = [C['aqua'] if v >= 4 else C['yellow'] if v == 3 else C['orange'] for v in score.values]
ax.barh(score.index, score.values, color=cols, height=0.6)
for i, v in enumerate(score.values):
    ax.text(v + 0.08, i, f'{v}/5', va='center', fontsize=11, fontweight='bold', color=INK2)
ax.set_xlim(0, 6.2); ax.set_xticks(range(0, 6))
ax.set_title('F1nalyze dataset scorecard (1 = poor, 5 = excellent)')
plt.tight_layout(); plt.show()
"""),

# ════════════════════════════════════════════════════════════════════════════
mc("sec11", r"""
## 11. Takeaways for modelling (bridge to the next presentation)

1. **Repair the data first** — `raceId_y == racerId`, 1 row per `resultId`.
2. **Target:** `positionOrder` in 4 classes; imbalance-robust metrics — **macro-F1**, balanced accuracy.
3. **Only features known before the start:** `grid`, `year`, `round`, age, nationality, circuit, team
   + **lagged historical features** (`shift(1)`): driver and team form, experience, points/wins before the race, DNF risk.
4. **Time-based validation only** (`TimeSeriesSplit`), never random.
5. **Baseline to beat:** the "class = starting position" rule — the charts in section 7.2 show it will be very strong.
6. **Raw driver IDs do not generalise** into the future (section 9) → instead of IDs: frequency encoding and historical statistics.

> **Modelling details:** notebook `02_modeling_classification.ipynb`.
"""),

cc("versions", r"""
import matplotlib, sys
print(f'Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | '
      f'matplotlib {matplotlib.__version__} | seaborn {sns.__version__}')
print('Notebook ran to completion ✔')
"""),

]  # end cells

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "00_dataset_overview.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Generated: {out} ({len(cells)} cells)")
