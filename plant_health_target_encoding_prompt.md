# Claude Code Prompt — Plant Health Prediction Project (Repo Setup + My Part Only)

Paste this entire prompt into Claude Code. It will create the full project repo structure and
generate ONLY my assigned preprocessing notebook (Categorical Target Encoding) as a fully
working, well-commented Jupyter notebook (.ipynb) using nbformat.

**IMPORTANT SCOPE NOTE:** The group is using 6 preprocessing techniques total (Standardization,
Categorical Target Encoding, Feature Selection via Mutual Information, Timestamp Feature
Engineering, Class Imbalance Handling, Outlier Detection via IQR). Do NOT implement any
technique other than Categorical Target Encoding — just create empty placeholder folders/files
for the other members so the repo structure is ready for them to fill in later. Each member's
notebook must be independent: it loads the RAW dataset directly and does not depend on any
other member's output file.

---

## PROJECT CONTEXT

- Task: Predict plant health status from sensor readings (multi-class classification)
- Dataset file: `plant_health_data.csv` (place at `data/raw/plant_health_data.csv`)
- Group: 6 members — each handles one independent preprocessing technique (and later, one model)
- My role: Categorical Target Encoding

---

## DATASET FACTS (use these — already inspected)

- Rows: 1,200 | Columns: 14 (no missing values)
- Columns: `Timestamp, Plant_ID, Soil_Moisture, Ambient_Temperature, Soil_Temperature, Humidity,
  Light_Intensity, Soil_pH, Nitrogen_Level, Phosphorus_Level, Potassium_Level,
  Chlorophyll_Content, Electrochemical_Signal, Plant_Health_Status`
- Target column: `Plant_Health_Status` — 3 classes: `Healthy` (299), `Moderate Stress` (401),
  `High Stress` (500) — classes are imbalanced (handled by another member, not me)
- `Plant_ID`: integer, 10 unique plants (1–10) — this is the categorical column to target-encode.
  Although stored as a number, it identifies a plant, not a measurable quantity.
- `Timestamp`: string datetime, e.g. `2024-10-03 10:54:53.407995` — feature engineering on this
  is handled by another member, not me. I only need to drop or ignore it for my own output.
- All other columns are continuous sensor readings (float).

---

## REPOSITORY STRUCTURE TO CREATE

```
Group_ID/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                                        ← place plant_health_data.csv here
│   └── external/
├── notebooks/
│   ├── IT0001_Standardization.ipynb                ← placeholder only, empty skeleton
│   ├── IT0002_CategoricalTargetEncoding.ipynb      ← MY NOTEBOOK — build fully
│   ├── IT0003_MutualInfoFeatureSelection.ipynb     ← placeholder only, empty skeleton
│   ├── IT0004_TimestampFeatureEngineering.ipynb    ← placeholder only, empty skeleton
│   ├── IT0005_ClassImbalanceHandling.ipynb         ← placeholder only, empty skeleton
│   └── IT0006_OutlierDetectionIQR.ipynb            ← placeholder only, empty skeleton
├── models/                                          ← empty, members add model notebooks later
└── results/
    ├── eda_visualizations/
    ├── logs/
    └── outputs/
```

For the 5 placeholder notebooks: create each with just a title markdown cell
(e.g. "# Member X — Standardization — TO BE COMPLETED BY ASSIGNED MEMBER") and one empty code
cell with `import pandas as pd`. Do not implement their logic.

---

## MY NOTEBOOK — notebooks/IT0002_CategoricalTargetEncoding.ipynb (build this fully)

**Section 1 — Introduction**
- Explain what target encoding is: replacing a categorical value with a statistic (e.g. mean
  or distribution) of the target variable for that category.
- Explain why it's useful here: `Plant_ID` is categorical (10 plants) but has real predictive
  signal (different plants may show different stress patterns) — target encoding captures that
  signal in a single numeric column instead of exploding into 10 one-hot columns.
- Note the target is multi-class (3 classes), so encoding must produce either one column per
  class (mean target-class probability per plant) or use a library that supports multi-class.

**Section 2 — Load Raw Data**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/plant_health_data.csv')
print(df.shape)
df.head()
```

**Section 3 — Basic Prep (only what's needed for this notebook)**
- Confirm no missing values with `df.isnull().sum()`.
- Drop the `Timestamp` column for this notebook only (note in markdown that timestamp feature
  engineering is a different member's task, out of scope here).
- Identify `Plant_ID` as the categorical column to encode; list all other numeric columns as-is.

**Section 4 — Explore the Categorical Column**
- Show `df['Plant_ID'].value_counts()`.
- Show a crosstab of `Plant_ID` vs `Plant_Health_Status` (counts and row-normalized
  percentages) to demonstrate that different plants have different health-status distributions.
- Plot a stacked/grouped bar chart of health status per plant.
  Save as `../results/eda_visualizations/M2_plantid_vs_healthstatus.png`

**Section 5 — Apply Categorical Target Encoding**
Since the target has 3 classes, implement it as **per-class mean target encoding** (also called
target-frequency / class-probability encoding): for each `Plant_ID`, compute the proportion of
rows belonging to each of the 3 health-status classes, and add those as 3 new numeric columns.

```python
# One-hot the target temporarily just to compute per-plant class proportions
target_dummies = pd.get_dummies(df['Plant_Health_Status'], prefix='target')
temp = pd.concat([df['Plant_ID'], target_dummies], axis=1)
plant_target_means = temp.groupby('Plant_ID').mean()
plant_target_means.columns = [
    f"PlantID_encoded_{c.replace('target_', '')}" for c in plant_target_means.columns
]
print(plant_target_means)

df_encoded = df.merge(plant_target_means, on='Plant_ID', how='left')
df_encoded.drop(columns=['Plant_ID'], inplace=True)
df_encoded.head()
```

- Explain in markdown: this gives each row 3 new columns representing how strongly its plant is
  historically associated with each health-status class, replacing the single `Plant_ID` column.
- Note the leakage risk: because we compute the encoding using the full dataset (including each
  row's own label), this can leak target information. Mention the proper fix (K-fold /
  leave-one-out target encoding) as a follow-up improvement, but keep the simple whole-dataset
  version for this coursework notebook since it's clearer to explain in the viva.

**Section 6 — Encode Target Variable for Downstream Use**
```python
target_map = {'Healthy': 0, 'Moderate Stress': 1, 'High Stress': 2}
df_encoded['Plant_Health_Status'] = df_encoded['Plant_Health_Status'].map(target_map)
```

**Section 7 — EDA Visualization**
Heatmap or bar chart comparing the 3 new `PlantID_encoded_*` columns across plants.
Title: "EDA: Target-Encoded Plant ID by Health Status".
Save as `../results/eda_visualizations/M2_EDA_target_encoding.png`

**Section 8 — Save Output**
```python
df_encoded.to_csv('../results/outputs/step_M2_categorical_target_encoded.csv', index=False)
print("Saved:", df_encoded.shape)
```
This output is independent — it does not depend on any other member's file, and no other
member's notebook should read from it either.

---

## FILE — README.md

Generate a README with:
- Project title: "Plant Health Status Prediction — Preprocessing Repo"
- Dataset description (1,200 rows, 14 columns, 3-class target, sensor readings from 10 plants)
- A table of all 6 preprocessing techniques and which notebook covers each, marking mine
  (`IT0002_CategoricalTargetEncoding.ipynb`) as complete and the rest as "TO BE COMPLETED"
- State clearly: each member's notebook is independent and loads the raw CSV directly — there
  is no dependency chain between notebooks
- How to run: install requirements, place `plant_health_data.csv` in `data/raw/`, then run
  `notebooks/IT0002_CategoricalTargetEncoding.ipynb`
- Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, nbformat

---

## FILE — requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
nbformat
jupyter
```

---

## FILE — .gitignore
```
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
Thumbs.db
.vscode/
.idea/
*.log
```
> NOTE: The dataset IS tracked by git (not ignored) so other members get it automatically.

---

## HOW TO RUN THIS PROMPT IN CLAUDE CODE

1. Create an empty folder on your PC for the project
2. Copy `plant_health_data.csv` into that folder (root — you can move it into `data/raw/`
   afterward, or ask Claude Code to move it)
3. Open Claude Code terminal inside that folder
4. Paste this entire prompt
5. Claude Code creates the full repo structure, 5 placeholder notebooks, and my complete
   Categorical Target Encoding notebook
6. Share the repo structure with your group so the other 5 members can fill in their own
   placeholder notebooks independently
