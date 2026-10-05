 # Plant Health Status Prediction — Preprocessing Repo

## Dataset

`data/raw/plant_health_data.csv` — 1,200 rows, 14 columns, no missing values. Sensor readings
from 10 plants (`Plant_ID` 1–10), each row timestamped, with a 3-class target
`Plant_Health_Status`:

| Class            | Count |
|------------------|-------|
| Healthy          | 299   |
| Moderate Stress  | 401   |
| High Stress      | 500   |

Columns: `Timestamp, Plant_ID, Soil_Moisture, Ambient_Temperature, Soil_Temperature, Humidity,
Light_Intensity, Soil_pH, Nitrogen_Level, Phosphorus_Level, Potassium_Level,
Chlorophyll_Content, Electrochemical_Signal, Plant_Health_Status`

## Preprocessing Techniques

The group is applying 6 independent preprocessing techniques, one per member. Each notebook
loads the **raw** dataset directly from `data/raw/plant_health_data.csv` — there is **no
dependency chain** between notebooks; no notebook reads another member's output file.

| Notebook                                          | Technique                                   | Status            |
|----------------------------------------------------|----------------------------------------------|-------------------|
| `IT0001_Standardization.ipynb`                     | Standardization                              | TO BE COMPLETED   |
| `IT0002_CategoricalTargetEncoding.ipynb`           | Categorical Target Encoding                  | ✅ Complete       |
| `IT0003_MutualInfoFeatureSelection.ipynb`          | Feature Selection via Mutual Information     | TO BE COMPLETED   |
| `IT0004_TimestampFeatureEngineering.ipynb`         | Timestamp Feature Engineering                | TO BE COMPLETED   |
| `IT0005_ClassImbalanceHandling.ipynb`              | Class Imbalance Handling                     | TO BE COMPLETED   |
| `IT0006_OutlierDetectionIQR.ipynb`                 | Outlier Detection via IQR                    | TO BE COMPLETED   |

## How to Run

1. Install requirements: `pip install -r requirements.txt`
2. Confirm `plant_health_data.csv` is at `data/raw/plant_health_data.csv` (already tracked in
   this repo).
3. Run `notebooks/IT0002_CategoricalTargetEncoding.ipynb`.

Outputs are written to `results/eda_visualizations/` (plots) and `results/outputs/`
(the encoded CSV: `step_M2_categorical_target_encoded.csv`).

## Requirements

pandas, numpy, matplotlib, seaborn, scikit-learn, nbformat
