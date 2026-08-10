# Data

The full study dataset is **not** included in this repository. It contains
physiological recordings collected under IRB approval, and redistributing
raw subject-level data is not something we can do here.

`sample_data.csv` is a small, already-scaled-compatible slice, three
subjects, a few hundred rows each, kept only so that:

- every script in this repository runs end to end out of the box,
- new contributors can sanity-check the pipeline without needing access to
  the private dataset, and
- CI / smoke tests have something real to run against.

It is **not** enough data to reproduce the numbers in the paper. To
retrain on the full dataset, request access as described in the paper and
point `configs/config.yaml`'s `data.unlabelled_csv` / `data.labelled_csv`
at your own copy. The expected schema is:

| Column | Meaning |
|---|---|
| `ACC_X, ACC_Y, ACC_Z` | Chest-worn accelerometer |
| `ECG` | Chest ECG |
| `EDA`, `Temp`, `Resp` | Chest EDA, skin temperature, respiration |
| `ACC_X_wrist, ACC_Y_wrist, ACC_Z_wrist` | Wrist accelerometer |
| `BVP`, `EDA_wrist`, `Temp_wrist` | Wrist BVP, EDA, skin temperature |
| `Stress_Class` | Integer label (0 = low, 1 = moderate, 2 = high) |
| `Task`, `SessionPhase`, `Activity` | Contextual signals, consumed by the context gate |
| `Subject_ID` | Used to window per subject and for subject-wise splits |

If your own data has different column names, update
`configs/config.yaml`'s `data.signal_columns` / `data.context_columns`
accordingly; nothing else in the codebase hardcodes column names.
