from joblib import load
import numpy as np
import pandas as pd

from . import Distribution as d
# import Distribution as d
# from prophet import Prophet  # or from fbprophet import Prophet
# import pandas as pd

def Load_Model(config_path = './model/expense_config_2.pkl'):
# Load trained models and config
    # models = joblib.load(model_path)   # dict: {"rf": ..., "gbr": ..., "ridge": ...}
    config = load(config_path)
    WINDOW = config["window"]
    HORIZON = config["horizon"]  # 30
    return WINDOW,HORIZON

def ensemble_predict_array(X_input):
    models = load('./model/expense_models_2.pkl')
    preds = np.column_stack([m.predict(X_input) for m in models.values()])
    return preds.mean(axis=1)

def normalize_d1(x):
    max1 = 999.0
    min1 = 0.9
    epsilon = 1e-6
    x = np.asarray(x, dtype=float)
    scaled = (x - min1) / (max1 - min1)
    scaled = np.clip(scaled, epsilon, 1 - epsilon)
    return scaled

def denormalize_d1(scaled):
    max1 = 999.0
    min1 = 0.9
    scaled = np.asarray(scaled, dtype=float)
    # assume scaled is already in [epsilon, 1-epsilon]
    x = scaled * (max1 - min1) + min1
    return x

def normalize_d2(x):
    max2 = 4996.0 
    min2 = 14.37
    epsilon = 1e-6
    x = np.asarray(x, dtype=float)
    scaled = (x - min2) / (max2 - min2)
    scaled = np.clip(scaled, epsilon, 1 - epsilon)
    return scaled

def denormalize_d2(scaled):
    max2 = 4996.0 
    min2 = 14.37
    scaled = np.asarray(scaled, dtype=float)
    x = scaled * (max2 - min2) + min2
    return x

def Choose_Normalization(chosen,data):
    if chosen == 'Dataset 1':
        return normalize_d1(data)
    else:
        return normalize_d2(data)
    

# def forecast_next_30_days(recent_values, recent_dates,WINDOW,HORIZON):
#     """
#     recent_values: list/array of last N expenses (N >= WINDOW)
#     recent_dates:  list/array of last N dates (strings or datetime), same length as recent_values
#     Returns: (forecast_value, future_dates)
#         - forecast_value: scalar, predicted average for next 30 days
#         - future_dates: pd.DatetimeIndex for those 30 days
#     """
#     if len(recent_values) < WINDOW:
#         raise ValueError(f"Need at least {WINDOW} recent values for forecasting")

#     # Take only the most recent WINDOW values
#     recent_values = np.array(recent_values[-WINDOW:], dtype=float)
#     recent_dates = pd.to_datetime(recent_dates[-WINDOW:])

#     # Build lag features: shape (1, WINDOW)
#     X_lag = recent_values.reshape(1, -1)

#     # Build calendar features using the "prediction anchor" date
#     # Here use the last observed date as the anchor
#     anchor_date = recent_dates[-1]
#     # For the ML model, we use calendar features at anchor time
#     X_cal = np.array([[anchor_date.dayofweek, anchor_date.month]], dtype=float)

#     # Combine lag + calendar (shape: (1, WINDOW + 2))
#     X_input = np.hstack([X_lag, X_cal])

#     # Predict average for next 30 days
#     avg_30 = ensemble_predict_array(X_input)[0]

#     # Build future date index for the next 30 days (if needed for UI)
#     future_dates = pd.date_range(start=anchor_date + pd.Timedelta(days=1),
#                                  periods=HORIZON, freq="D")

#     return avg_30, future_dates

def forecast_next_30_days(recent_values, recent_dates, window=60, horizon=30):
    if len(recent_values) < window:
        return None, None

    vals = np.array(recent_values[-window:], dtype=float)
    dates = pd.to_datetime(recent_dates[-window:])

    # lag features (in same scale as training)
    X_lag = vals.reshape(1, -1)

    # rolling stats (use same choice as in training)
    v = vals
    roll_7  = v[-7:].mean()  if len(v) >= 7  else v.mean()
    roll_14 = v[-14:].mean() if len(v) >= 14 else v.mean()
    roll_30 = v[-30:].mean() if len(v) >= 30 else v.mean()

    std_7  = v[-7:].std(ddof=1)  if len(v) >= 7  else v.std(ddof=1)
    std_14 = v[-14:].std(ddof=1) if len(v) >= 14 else v.std(ddof=1)
    std_30 = v[-30:].std(ddof=1) if len(v) >= 30 else v.std(ddof=1)

    anchor_date = dates[-1]
    dayofweek  = anchor_date.dayofweek
    month      = anchor_date.month
    dayofmonth = anchor_date.day
    is_weekend = 1 if dayofweek >= 5 else 0
    is_payday  = 1 if dayofmonth in [1, 30, 31] else 0

    # EXACT same order and count as training
    X_cal = np.array([[dayofweek, month, dayofmonth,
                       is_weekend, is_payday,
                       roll_7, roll_14, roll_30,
                       std_7, std_14, std_30]], dtype=float)

    X_input = np.hstack([X_lag, X_cal])  # shape (1, 60 + 11 = 71)

    avg_30_norm = ensemble_predict_array(X_input)[0]

    future_dates = pd.date_range(
        start=anchor_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    return avg_30_norm, future_dates


def denormalize_results(chosen,data):
    if chosen == 'Dataset 1':
            avg_next_30 = denormalize_d1(data)
    else:
            avg_next_30 = denormalize_d2(data)
        # print("\nPredicted avg spend next 30 days:", avg_next_30)
        # print("Future dates (next 30 days anchor-based):")
    print(avg_next_30)
    return avg_next_30
    # print(future_dates)


def Predict_Model(normalzized_data,user_dates,WINDOW,HORIZON,chosen):
    WINDOW,HORIZON = Load_Model()
    avg_next_30, future_dates = forecast_next_30_days(normalzized_data,user_dates,WINDOW, HORIZON)

    if avg_next_30 is None:
        print("Not enough data yet (cold start).")
    else:
        print("\nPredicted avg spend next 30 days:", avg_next_30)
        print("Future dates (next 30 days anchor-based):")
        print(future_dates)
    
    return denormalize_results(chosen=chosen)

def Predict(user_expenses,user_dates):
    data1,data2 = d.Distributions()
    print (data1,data2)
    chosen = d.Hypotesis_testing(user_expenses,data1,data2)
    if chosen == 'Dataset 1':
        normalized_values = normalize_d1(user_expenses)
    else:
        normalized_values = normalize_d2(user_expenses)
    
    print(normalized_values)
    WINDOW,HORIZON = Load_Model()
    avg_next_30, future_dates = forecast_next_30_days(normalized_values, user_dates,WINDOW, HORIZON)
    if avg_next_30 is None:
        print("Not enough data yet (cold start).")
    else:
        print("\nPredicted avg spend next 30 days:", avg_next_30)
        print("Future dates (next 30 days anchor-based):")
        print(future_dates)

    y = denormalize_results(chosen,data=avg_next_30)
    print(y)
    return y

# def Generate_sample(n):
#     np.random.seed(42)
#     size = n
#     print(size)

#     # Example: mostly low-range spends with some noise
#     user_expenses = np.random.uniform(200, 1500, size=size).round(2).tolist()
#     print('2')
#     # -----------------------------
#     # 2. Create 60 dates with gaps
#     # -----------------------------
#     start = pd.Timestamp("2025-01-01")
#     dates = []

#     current = start
#     for i in range(size):
#         dates.append(current)
#         # Random gap: 1–3 days
#         gap = np.random.choice([1, 1, 2, 3])   # mostly 1, sometimes 2–3
#         current = current + pd.Timedelta(days=gap)

#     user_dates = [d.strftime("%Y-%m-%d") for d in dates]

#     print("Number of expenses:", len(user_expenses))
#     print("Number of dates:", len(user_dates))
#     print("First 5:")

#     return user_expenses[:-1],user_dates[:-1]

def Generate_sample(n, p1=0.4):
    np.random.seed(42)
    data1, data2 = d.Distributions()

    n1 = int(n * p1)
    n2 = n - n1

    idx1 = np.random.randint(0, len(data1), size=n1)
    idx2 = np.random.randint(0, len(data2), size=n2)

    samples = np.concatenate([data1[idx1], data2[idx2]])
    np.random.shuffle(samples)

    user_expenses = samples.round(2).tolist()

    start = pd.Timestamp("2025-01-01")
    dates = []
    current = start
    for _ in range(n):
        dates.append(current)
        gap = np.random.choice([1, 1, 2, 3])
        current = current + pd.Timedelta(days=gap)

    user_dates = [d.strftime("%Y-%m-%d") for d in dates]

    return user_expenses[:-1], user_dates[:-1]

if __name__ == '__main__':

        # models,config,WINDOW,HORIZON = Load_Model()
        # Predict_Model(model=model)
    np.random.seed(42)
    size = 1200

    # Example: mostly low-range spends with some noise
    user_expenses = np.random.uniform(200, 1500, size=size).round(2).tolist()

    # -----------------------------
    # 2. Create 60 dates with gaps
    # -----------------------------
    start = pd.Timestamp("2025-01-01")
    dates = []

    current = start
    for i in range(size):
        dates.append(current)
        # Random gap: 1–3 days
        gap = np.random.choice([1, 1, 2, 3])   # mostly 1, sometimes 2–3
        current = current + pd.Timedelta(days=gap)

    user_dates = [d.strftime("%Y-%m-%d") for d in dates]

    print("Number of expenses:", len(user_expenses))
    print("Number of dates:", len(user_dates))
    print("First 5:")
    # for e, d in list(zip(user_expenses, user_dates))[:5]:
        # print(d, "->", e)

    # print("Last 5:")
    # for e, d in list(zip(user_expenses, user_dates))[-5:]:
        # print(d, "->", e)
    Predict(user_expenses,user_dates)