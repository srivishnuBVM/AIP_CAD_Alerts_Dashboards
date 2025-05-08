# preprocess_and_save.py
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
import re

def build_next_year_features(df_wide, feature_names, current_year=2024):
    data = {}
    pattern = re.compile(r'_(\d{4})$')
    year_suffix = f'_{current_year}'
    excluded_features = {'SCHOOL_NAME', 'DISTRICT_NAME'}

    for feat in feature_names:
        if feat in excluded_features:
            continue
        m = pattern.search(feat)
        if m:
            this_year_col = pattern.sub(year_suffix, feat)
            data[feat] = (
                df_wide[this_year_col]
                if this_year_col in df_wide.columns
                else df_wide[feat]
            )
        else:
            data[feat] = df_wide[feat]
    return pd.DataFrame(data, index=df_wide.index)

# Load and preprocess
df = pd.read_excel("AttendanceDataFinal.xlsx", engine="openpyxl")
model = XGBRegressor()
model.load_model("regressor_model.xgb")
model2 = XGBClassifier()
model2.load_model("final_model_latest.xgb")

X_2025 = build_next_year_features(df, model.get_booster().feature_names, current_year=2024)
df["Predicted_2025"] = model.predict(X_2025)
df["Probabilities_2025"] = model2.predict_proba(X_2025)[:, 1]

# Save to optimized format
df.to_parquet("AttendanceDataProcessed.parquet")
