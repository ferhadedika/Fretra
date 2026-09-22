import joblib, pathlib, pandas as pd, numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

SCRIPT_DIR    = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent / "model_artifacts"

df = pd.read_csv(SCRIPT_DIR / "scfi_final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df = df.rename(columns={
    "Date": "date", "Brent_Crude": "Brent_Oil",
    "PMI_Manufacturing": "World_PMI",
    "Q_2":"Q2","Q_3":"Q3","Q_4":"Q4",
    "Month_2":"Month_Feb","Month_3":"Month_Mar","Month_4":"Month_Apr",
    "Month_5":"Month_May","Month_6":"Month_Jun","Month_7":"Month_Jul",
    "Month_8":"Month_Aug","Month_9":"Month_Sep","Month_10":"Month_Oct",
    "Month_11":"Month_Nov","Month_12":"Month_Dec",
})
df = df.dropna().reset_index(drop=True)

EXOG_FEATURES = [
    "Brent_Oil","CNY_USD","World_PMI","Trade_Volume_Index","Trade_Volume_YoY",
    "Q2","Q3","Q4",
    "Month_Feb","Month_Mar","Month_Apr","Month_May","Month_Jun",
    "Month_Jul","Month_Aug","Month_Sep","Month_Oct","Month_Nov","Month_Dec",
]

y_s = df["SCFI"].values
X_s = df[EXOG_FEATURES].values

print("Fitting SARIMAX...")
res = SARIMAX(
    y_s, exog=X_s, order=(1,1,1), seasonal_order=(0,0,0,0),
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False, maxiter=500, method="powell")

res.save(str(ARTIFACTS_DIR / "sarimax.pkl"))
print("Saved → sarimax.pkl")

joblib.dump(X_s[-50:], ARTIFACTS_DIR / "sarimax_exog_background.pkl")
print("Saved → sarimax_exog_background.pkl")