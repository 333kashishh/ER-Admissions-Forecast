"""
generate_data.py
Creates a realistic synthetic dataset of daily ER admissions for a large
city hospital (Delhi-like climate) from 2023-01-01 to 2025-12-31.

Why synthetic? Real hospital admission records are private patient data.
This script builds data whose patterns match published findings:
  - Mondays are the busiest ER day, weekends the quietest
  - Heatwaves (May-June) and cold waves (Dec-Jan) raise admissions
  - Poor air quality (AQI) raises respiratory/cardiac visits
  - Flu season (Jan-Feb and the monsoon, Jul-Sep in India) raises visits
Run:  python generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # fixed seed -> same data every run

dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
n = len(dates)
month = dates.month.values
doy = dates.dayofyear.values

# ---------------- Air Quality Index ----------------
# Delhi pattern: very poor in winter (Nov-Jan), cleaner in monsoon (Jul-Sep)
aqi_season = 210 + 140 * np.cos(2 * np.pi * (doy - 340) / 365)
aqi = aqi_season + rng.normal(0, 45, n)
aqi = np.clip(aqi, 35, 490).round().astype(int)

# ---------------- Extreme weather alerts ----------------
weather = np.array(["No Alert"] * n, dtype=object)
heat_prob = np.where(np.isin(month, [5, 6]), 0.35, np.where(month == 4, 0.08, 0.0))
cold_prob = np.where(np.isin(month, [12, 1]), 0.25, 0.0)
u = rng.random(n)
weather[u < heat_prob] = "Heatwave"
weather[(u < cold_prob) & (weather == "No Alert")] = "Coldwave"

# ---------------- Flu season ----------------
flu = np.isin(month, [1, 2, 7, 8, 9]).astype(int)

# ---------------- Day of week ----------------
dow = dates.day_name().values
dow_effect = {"Monday": 0.12, "Tuesday": 0.05, "Wednesday": 0.02,
              "Thursday": 0.0, "Friday": 0.01, "Saturday": -0.06, "Sunday": -0.10}

# ---------------- Expected admissions (log-linear) ----------------
log_mu = (np.log(115)
          + np.vectorize(dow_effect.get)(dow)
          + 0.0009 * (aqi - 200)                 # +~9% per 100 AQI points
          + np.where(weather == "Heatwave", 0.20, 0)
          + np.where(weather == "Coldwave", 0.13, 0)
          + 0.15 * flu
          + 0.00012 * np.arange(n))              # slow growth in city population
mu = np.exp(log_mu)

# Extra day-to-day randomness (overdispersion): gamma-Poisson = Negative Binomial
shape = 60
admissions = rng.poisson(rng.gamma(shape, mu / shape))

df = pd.DataFrame({
    "date": dates.date,
    "day_of_week": dow,
    "weather_alert": weather,
    "aqi": aqi,
    "flu_season": np.where(flu == 1, "Yes", "No"),
    "er_admissions": admissions,
})

# Real-world data is messy: introduce a few missing AQI readings (sensor outages)
missing_idx = rng.choice(n, size=18, replace=False)
df.loc[missing_idx, "aqi"] = np.nan

df.to_csv("data/er_admissions.csv", index=False)
print(f"Saved {len(df)} rows to data/er_admissions.csv")
print(df.head())
