# # app.py
# import streamlit as st
# import numpy as np
# import pandas as pd

# from Categorization.categorizer import Categorization
# import Forecasting.Model as md
# from Forecasting.conversion import city

# st.set_page_config(
#     page_title="Smart Expense Categorizer & Forecaster",
#     page_icon="💸",
#     layout="wide",
# )

# # ---------------- Sidebar: global controls ----------------
# st.sidebar.title("Global settings")

# city_options = {
#     "Boston, United States": ("Boston", "United States"),
#     "Mumbai, India": ("Mumbai", "India"),
#     "Dubai, UAE": ("Dubai", "United Arab Emirates"),
#     "London, United Kingdom": ("London", "United Kingdom"),
#     "Singapore, Singapore": ("Singapore", "Singapore"),
# }
# selected_city_label = st.sidebar.selectbox("City", list(city_options.keys()))
# city_name, country_name = city_options[selected_city_label]

# currency_options = ["INR", "USD", "AED", "EUR", "GBP"]
# output_currency = st.sidebar.selectbox("Output currency", currency_options, index=1)
# model_currency = "INR"

# st.sidebar.markdown("---")
# st.sidebar.caption("Demo: NLP + time‑series forecasting for personal expenses.")

# # ---------------- Tabs for pages ----------------
# tab_home, tab_cat, tab_forecast = st.tabs(["🏠 Home", "🧾 Categorization", "📈 Forecasting"])

# # ---------------- Home tab ----------------
# with tab_home:
#     st.title("Smart Expense Categorizer & Forecaster")

#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown(
#             """
#             This demo shows how an NLP expense categorizer (Sentence‑BERT + ML)
#             and a time‑series forecaster can work together to analyze your spending.
#             """
#         )
#         st.markdown(
#             """
#             **What you can do here:**
#             - Type any expense description and see its predicted category.
#             - Run a monthly expense forecast, scaled to different cities and currencies.
#             - Inspect intermediate outputs, such as model inputs and raw predictions.
#             """
#         )
#     with col2:
#         st.info(
#             f"Current demo location: **{city_name}, {country_name}**\n\n"
#             f"Forecasts are shown in **{model_currency}** and converted to **{output_currency}**."
#         )

# # ---------------- Categorization tab ----------------
# with tab_cat:
#     st.header("Expense categorization")

#     user_text = st.text_input("Enter expense description", value="Uber ride last night")

#     col_main, col_side = st.columns([2, 1])

#     with col_main:
#         if st.button("Predict category", key="cat_button"):
#             with st.spinner("Running NLP model to categorize your expense..."):
#                 category = Categorization(user_text)

#             st.success(f"Predicted category: **{category}**")

#             with st.expander("Show intermediate NLP details"):
#                 st.markdown(
#                     """
#                     - Raw input text.
#                     - Encoded sentence embedding (from Sentence‑BERT).
#                     - Ensemble model prediction (SVM + XGBoost probabilities).
#                     """
#                 )
#                 st.write({"input_text": user_text})
#                 st.caption("You can plug in and display actual embeddings / probas here.")

#     with col_side:
#         st.subheader("How it works")
#         st.write(
#             """
#             The description is embedded using a sentence transformer
#             and passed to an ensemble classifier to predict a spending category.
#             """
#         )

# # ---------------- Forecasting tab ----------------
# with tab_forecast:
#     st.header("Monthly expense forecasting")

#     st.markdown(
#         f"Forecast is generated in **{model_currency}**, then scaled to "
#         f"**{city_name}, {country_name}** and converted to **{output_currency}**."
#     )

#     if st.button("Run forecast", key="forecast_button"):
#         with st.spinner("Running forecasting model and scaling to selected city..."):
#             # model + scaling pipeline
#             scale = city(city_name=city_name, country_name=country_name, output_currency=output_currency)

#             user_expenses, user_dates = md.Generate_sample()
#             avg = md.Predict(user_expenses, user_dates)  # your current API
#             forecast_mumbai_inr = float(avg) * 30  # simple example
#             forecast_city_inr = forecast_mumbai_inr * scale

#             df_forecast = pd.DataFrame(
#                 {
#                     "date": [user_dates[-1]],
#                     f"forecast_{city_name}_{model_currency}": [forecast_mumbai_inr],
#                     f"forecast_{city_name}_{output_currency}": [forecast_city_inr],
#                 }
#             )

#         st.subheader("Forecast table")
#         st.dataframe(df_forecast)

#         st.subheader("Forecast plot")
#         st.line_chart(df_forecast.set_index("date"))

#         with st.expander("Show intermediate forecasting details"):
#             st.markdown(
#                 """
#                 - Sampled user expense history (last few records).
#                 - Model raw output (per‑day average in INR).
#                 - Applied location scale factor and currency conversion.
#                 """
#             )
#             st.write("Sample user expenses:", user_expenses[-5:])
#             st.write("Last dates:", user_dates[-5:])
#             st.write(
#                 {
#                     "avg_daily_inr": float(avg),
#                     "monthly_forecast_inr": forecast_mumbai_inr,
#                     "scale_factor": scale,
#                     "monthly_forecast_scaled": forecast_city_inr,
#                 }
#             )
# app.py
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import streamlit as st

from Categorization.categorizer import Categorization
import Forecasting.Model as md
from Forecasting.conversion import city
from Forecasting.Distribution import Distributions
from Optimization.optimize import optimize_answer

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Mint Sage",
    page_icon="💸",
    layout="wide",
)

# ---------------- Helpers ----------------
def system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    return cpu, mem

def plot_beta_fit_three(data1, data2, sample_data, title: str):
    # Convert to numpy
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    sample_data = np.asarray(sample_data, dtype=float)

    # Guard against empty / constant data
    if (
        data1.size == 0
        or data2.size == 0
        or sample_data.size == 0
        or np.all(sample_data == sample_data[0])
    ):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Not enough variation / data to plot", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Normalize all three to (0,1) using global min/max so scales match
    all_data = np.concatenate([data1, data2, sample_data])
    dmin, dmax = all_data.min(), all_data.max()
    eps = 1e-9
    data1_n = (data1 - dmin) / (dmax - dmin + eps)
    data2_n = (data2 - dmin) / (dmax - dmin + eps)
    sample_n = (sample_data - dmin) / (dmax - dmin + eps)

    # Clip into open interval (0,1) to satisfy beta MLE requirements
    clip_eps = 1e-6
    data1_n = np.clip(data1_n, clip_eps, 1.0 - clip_eps)
    data2_n = np.clip(data2_n, clip_eps, 1.0 - clip_eps)
    sample_n = np.clip(sample_n, clip_eps, 1.0 - clip_eps)

    # Fit beta for data1 and data2
    a1, b1, _, _ = stats.beta.fit(data1_n, floc=0, fscale=1)
    a2, b2, _, _ = stats.beta.fit(data2_n, floc=0, fscale=1)

    x = np.linspace(0, 1, 200)
    pdf1 = stats.beta.pdf(x, a1, b1, loc=0, scale=1)
    pdf2 = stats.beta.pdf(x, a2, b2, loc=0, scale=1)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Histogram of sample data only
    ax.hist(
        sample_n,
        bins=20,
        density=True,
        alpha=0.4,
        label="Sample data",
        color="tab:gray",
    )

    # Overlay beta curves for data1 and data2
    ax.plot(x, pdf1, "r-", lw=2, label=f"Data 1 beta (a={a1:.2f}, b={b1:.2f})")
    ax.plot(x, pdf2, "b-", lw=2, label=f"Data 2 beta (a={a2:.2f}, b={b2:.2f})")

    ax.set_xlabel("Normalized value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return fig

def aggregate_monthly_stats(dates, values):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": values})
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["y"].agg(
        monthly_sum="sum",
        count_days="count",
    ).reset_index()
    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly["avg_daily"] = monthly["monthly_sum"] / monthly["count_days"]
    monthly["is_forecast"] = False
    return monthly[["date", "monthly_sum", "avg_daily", "is_forecast"]]

def aggregate_rolling_30(dates, values):
    """Compute rolling 30-day average and sum; index at last date of each 30-day window."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": values}).sort_values("date")
    df.set_index("date", inplace=True)
    # 30D calendar window; adjust '30D' if your horizon differs.[web:181][web:186]
    roll_sum = df["y"].rolling("30D").sum()
    roll_count = df["y"].rolling("30D").count()
    roll_avg = roll_sum / roll_count
    out = pd.DataFrame(
        {
            "date": roll_sum.index,
            "window_sum": roll_sum.values,
            "avg_daily": roll_avg.values,
        }
    ).dropna()
    out["is_forecast"] = False
    return out




# ---------------- Sidebar: global settings ----------------
st.sidebar.title("Global settings")

city_options = {
    "Boston, United States": ("Boston", "United States"),
    "Mumbai, India": ("Mumbai", "India"),
    "Dubai, UAE": ("Dubai", "United Arab Emirates"),
    "London, United Kingdom": ("London", "United Kingdom"),
    "Singapore, Singapore": ("Singapore", "Singapore"),
}
selected_city_label = st.sidebar.selectbox("City", list(city_options.keys()))
city_name, country_name = city_options[selected_city_label]

currency_options = ["INR", "USD", "AED", "EUR", "GBP"]
output_currency = st.sidebar.selectbox("Output currency", currency_options, index=1)
model_currency = "INR"

st.sidebar.markdown("---")
st.sidebar.caption("Demo: NLP + time‑series forecasting for personal expenses.")

st.sidebar.markdown("### Logs")

cat_log_container = st.sidebar.container()
fc_log_container = st.sidebar.container()

def log_cat(msg: str):
    cat_log_container.write(f"[Categorization] {msg}")

def log_fc(msg: str):
    fc_log_container.write(f"[Forecasting] {msg}")

# ---------------- Tabs ----------------
tab_home, tab_cat, tab_forecast, tab_opt = st.tabs(
    ["🏠 Home", "🧾 Categorization", "📈 Forecasting", "🧠 Optimization"]
)

# # ---------------- Home tab ----------------
# with tab_home:
#     st.title("Smart Expense Categorizer & Forecaster")

#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown(
#             """
#             This demo combines an NLP-based expense categorizer with a time‑series
#             forecaster to help you understand and project your spending.
#             """
#         )
#         st.markdown(
#             """
#             **You can:**
#             - Enter an expense description and view its predicted category.
#             - Run a monthly expense forecast for a chosen city and currency.
#             - Inspect logs, resource usage, probabilities, and fitted distributions.
#             """
#         )
#     with col2:
#         st.info(
#             f"Current demo location: **{city_name}, {country_name}**\n\n"
#             f"Forecasts are shown in **{model_currency}** and converted to "
#             f"**{output_currency}**."
#         )

with tab_home:
    st.title("Mint Sage – Smart Expense Categorizer & Forecaster")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            **Mint Sage** is an AI-based expense tracker, forecaster, and optimizer
            that helps you understand where your money goes and how to improve your spending habits.
            """
        )
        st.markdown(
            """
            ### What this demo showcases
            - **NLP categorization:** MiniLM (Sentence-BERT) + SVM/XGBoost ensemble to map raw transaction text into categories like Food, Transport, Utilities, etc.
            - **Time-series forecasting:** Sliding 60-day window with rolling stats and calendar features to predict the average expense over the next 30 days.
            - **Probability modeling:** Two beta-like expense distributions combined via statistical modeling and hypothesis testing for robust normalization of unseen sequence.
            """
        )
        with st.expander("Behind the scenes"):
            st.markdown(
                """
                - **Data sources:** Two Kaggle finance datasets in INR plus a labeled dataset for categorization.
                - **Baes Models tested:** Exponential Smoothing, SARIMA, Prophet, Random Forest, Gradient Boosting, Ridge, and an ensemble tuned on MAE.
                - **Optimization:** LLM-powered optimization agent for budget advice, subscription cleanup, and goal-based planning.
                """
            )

    with col2:
        st.markdown("### Project snapshot")
        st.info(
            f"Current demo location: **{city_name}, {country_name}**\n\n"
            f"Forecasts are computed in **{model_currency}** and converted to "
            f"**{output_currency}** based on your selection."
        )
        st.markdown(
            """
            **Team 104**
            - Viraj: EDA, LLM integration, APIs, frontend, Plaid integration.[file:193]
            - Vedant: Distribution modeling, hypothesis testing, forecasting module.[file:193]
            - Archita: Categorization engine (TF‑IDF → USE → MiniLM), SVM+XGBoost ensemble.[file:193]
            """
        )


# ---------------- Categorization tab ----------------
# with tab_cat:
#     st.header("Expense categorization")

#     user_text = st.text_input(
#         "Enter expense description", value="Uber ride last night"
#     )

#     col_main, col_side = st.columns([2, 1])

#     with col_main:
#         if st.button("Predict category", key="cat_button"):
#             log_cat(f"Received text: {user_text}")
#             cpu_before, mem_before = system_metrics()
#             t0 = time.time()

#             with st.spinner("Running NLP model to categorize your expense..."):
#                 # Expect Categorization to return (label, prob_array, class_labels)
#                 # Adapt this line to your actual function signature.
#                 category, probs, class_labels = Categorization(user_text)
#                 time.sleep(0.1)  # just to make spinner visible

#             dt = time.time() - t0
#             cpu_after, mem_after = system_metrics()
#             log_cat(f"Predicted: {category} in {dt:.3f}s")

#             st.success(f"Predicted category: **{category}**")

#             # Probabilities
#             if probs is not None and class_labels is not None:
#                 st.subheader("Categorization probabilities")
#                 prob_df = pd.DataFrame(
#                     {"class": class_labels, "probability": probs[0]}
#                 )
#                 st.bar_chart(prob_df.set_index("class"))

#             # Metrics
#             st.subheader("Run metrics")
#             m1, m2, m3 = st.columns(3)
#             m1.metric("Latency (s)", f"{dt:.3f}")
#             m2.metric("CPU usage (%)", f"{cpu_after}")
#             m3.metric("Memory usage (%)", f"{mem_after}")

#             # Intermediate details
#             with st.expander("Show intermediate NLP details"):
#                 st.write({"input_text": user_text})
#                 st.caption(
#                     "You can hook in and display actual embeddings / model "
#                     "internals from the backend here."
#                 )

#     with col_side:
#         st.subheader("How it works")
#         st.write(
#             """
#             - Text is embedded using a sentence transformer (Sentence‑BERT).\n
#             - Embedding is passed into an ensemble classifier (e.g., SVM + XGBoost).\n
#             - Final probabilities are combined and the top class is selected.
#             """
#         )
with tab_cat:
    st.header("Expense categorization")

    user_text = st.text_input(
        "Enter expense description", value="Uber ride last night"
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        if st.button("Predict category", key="cat_button"):
            log_cat(f"Received text: {user_text}")
            cpu_before, mem_before = system_metrics()
            t0 = time.time()

            with st.spinner("Running NLP model to categorize your expense..."):
                category, probs, class_labels = Categorization(user_text)
                time.sleep(0.1)

            dt = time.time() - t0
            cpu_after, mem_after = system_metrics()
            log_cat(f"Predicted: {category} in {dt:.3f}s")

            st.success(f"Predicted category: **{category}**")

            if probs is not None and class_labels is not None:
                probs = np.asarray(probs, dtype=float).ravel()
                class_labels = list(class_labels)  # these are now strings

                if len(probs) == len(class_labels):
                    st.subheader("Categorization probabilities")
                    prob_df = pd.DataFrame(
                        {"class": class_labels, "probability": probs}
                    )
                    st.bar_chart(prob_df.set_index("class"))
                else:
                    st.warning(
                        f"Length mismatch: got {len(probs)} probabilities "
                        f"for {len(class_labels)} classes. Not plotting."
                    )



            st.subheader("Run metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency (s)", f"{dt:.3f}")
            m2.metric("CPU usage (%)", f"{cpu_after}")
            m3.metric("Memory usage (%)", f"{mem_after}")

            with st.expander("Show intermediate NLP details"):
                st.write({"input_text": user_text})
                st.caption(
                    "You can hook in and display actual embeddings / model "
                    "internals from the backend here."
                )


with tab_forecast:
    st.header("Monthly expense forecasting")

    st.markdown(
        f"Forecast is generated in **{model_currency}**, then scaled to "
        f"**{city_name}, {country_name}** and converted to **{output_currency}**."
    )

    source = st.radio(
        "Choose data source",
        ["Generate synthetic data", "Upload CSV with ds,y"],
        index=0,
        key="forecast_source",
    )

    n_samples = None
    uploaded_file = None

    if source == "Generate synthetic data":
        n_samples = st.slider(
            "Number of samples to generate",
            min_value=60,
            max_value=365,
            value=120,
            step=10,
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV with columns 'ds' (date) and 'y' (values). Minimum 60 rows.",
            type="csv",
            key="forecast_csv",
        )

    if st.button("Run forecast", key="forecast_button"):
        log_fc("Starting forecast pipeline")
        cpu_before, mem_before = system_metrics()
        t0 = time.time()

        # --------- Build time series (user_expenses, user_dates) ----------
        if source == "Generate synthetic data":
            user_expenses, user_dates = md.Generate_sample(n=n_samples)
            log_fc(f"Generated {len(user_expenses)} synthetic samples")
        else:
            if uploaded_file is None:
                st.error("Please upload a CSV file first.")
                st.stop()

            df_in = pd.read_csv(uploaded_file)
            if not {"ds", "y"}.issubset(df_in.columns):
                st.error("CSV must contain columns 'ds' and 'y'.")
                st.stop()

            if len(df_in) < 60:
                st.error("CSV must have at least 60 rows.")
                st.stop()

            df_in = df_in.sort_values("ds")
            user_dates = pd.to_datetime(df_in["ds"]).tolist()
            user_expenses = df_in["y"].astype(float).tolist()
            log_fc(f"Loaded {len(user_expenses)} rows from uploaded CSV")

        with st.spinner("Running forecasting model and scaling to selected city..."):
            scale = city(
                city_name=city_name,
                country_name=country_name,
                output_currency=output_currency,
            )

            # model: avg daily spend over next 30 days (already denormalized in Predict)
            avg_daily_inr = float(md.Predict(user_expenses, user_dates))

            # Historical sequence (daily)
            hist_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(user_dates),
                    f"historical_{model_currency}": user_expenses,
                }
            )

            # Rolling 30-day history ending at each date
            rolling_hist = aggregate_rolling_30(
                hist_df["date"], hist_df[f"historical_{model_currency}"]
            )

            # Forecast point: same 30-day window but shifted forward
            last_date = pd.to_datetime(user_dates[-1])
            horizon_days = 30
            forecast_end = last_date + pd.Timedelta(days=horizon_days)

            forecast_window_sum = avg_daily_inr * horizon_days
            forecast_city_inr = forecast_window_sum * scale / horizon_days  # keep your scale rule

            df_forecast = pd.DataFrame(
                {
                    "date": [forecast_end],
                    f"forecast {model_currency} Monthly": [forecast_window_sum],
                    f"forecast {model_currency} Daily Average": [avg_daily_inr],
                    f"forecast {city_name} {output_currency} Next Month": [forecast_city_inr],
                }
            )

            rolling_forecast = pd.DataFrame(
                {
                    "date": [forecast_end],
                    "window_sum": [forecast_window_sum],
                    "avg_daily": [avg_daily_inr],
                    "is_forecast": [True],
                }
            )

            # Distributions for beta plots
            data1, data2 = Distributions()
            sample_data = np.asarray(user_expenses, dtype=float)

            time.sleep(0.1)

        dt = time.time() - t0
        cpu_after, mem_after = system_metrics()
        log_fc(
            f"avg_daily_inr={avg_daily_inr:.2f}, scale={scale:.3f}, "
            f"done in {dt:.3f}s"
        )

        # 1) Distribution plots
        st.subheader("Distribution fits (beta)")
        fig = plot_beta_fit_three(
            data1, data2, sample_data,
            "Data1 & Data2 beta fits + sample histogram",
        )
        st.pyplot(fig)

        # 2) Forecast result (30‑day window view)
        st.subheader("Next 30‑day window forecast (anchor-based)")
        st.dataframe(df_forecast)

        # Combine rolling history + forecast, indexed by window end date
        rolling_combined = pd.concat(
            [rolling_hist, rolling_forecast], ignore_index=True
        )
        rolling_combined["date"] = pd.to_datetime(rolling_combined["date"])
        rolling_combined = rolling_combined.sort_values("date").reset_index(drop=True)

        display_df = rolling_combined.copy()
        display_df["type"] = np.where(display_df["is_forecast"], "Forecast", "History")

        st.subheader("30‑day rolling windows (history + forecast)")
        st.dataframe(display_df.set_index("date"))

        # Avg daily cost per 30‑day window
        st.subheader("Avg daily cost per 30‑day window (INR)")
        st.line_chart(
            rolling_combined.set_index("date")[["avg_daily"]]
        )

        # Total cost per 30‑day window
        st.subheader("Total cost per 30‑day window (INR)")
        st.line_chart(
            rolling_combined.set_index("date")[["window_sum"]]
        )

        # Metrics
        st.subheader("Run metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Latency (s)", f"{dt:.3f}")
        m2.metric("CPU usage (%)", f"{cpu_after}")
        m3.metric("Memory usage (%)", f"{mem_after}")

with tab_opt:
    st.header("Optimization assistant")

    st.markdown(
        """
        This section combines your **historical expenses**, their **categories**,
        and the **latest forecast** into a single view, then lets you ask
        free-form questions about how to optimize your spending.
        """
    )

    # --- Data source controls for optimization DF ---
    opt_source = st.radio(
        "Choose optimization data source",
        ["Generate sample data", "Upload CSV (date, expense, category, is_forecast)"],
        index=0,
        key="opt_source",
    )

    opt_uploaded = None
    if opt_source == "Upload CSV (date, expense, category, is_forecast)":
        opt_uploaded = st.file_uploader(
            "Upload CSV with columns: date, expense, category, is_forecast",
            type="csv",
            key="opt_csv",
        )

    # Build / load opt_df
    if opt_source == "Generate sample data":
        # Generate synthetic optimization dataframe
        if "opt_df" not in st.session_state or st.button("Regenerate sample data", key="regen_opt"):
            np.random.seed(0)
            sample_dates = pd.date_range("2025-01-01", periods=90, freq="D")
            sample_exp = np.random.uniform(200, 1500, size=90).round(2)
            sample_cat = np.random.choice(
                ["Food", "Transport", "Shopping", "Utilities", "Health"],
                size=90,
            )
            is_fc = [False] * 89 + [True]  # last row as forecast marker
            st.session_state.opt_df = pd.DataFrame(
                {
                    "date": sample_dates,
                    "expense": sample_exp,
                    "category": sample_cat,
                    "is_forecast": is_fc,
                }
            )
        opt_df = st.session_state.opt_df

    else:
        if opt_uploaded is None:
            st.warning("Upload a CSV to use it in the optimizer, or switch to generated data.")
            st.stop()

        df_in = pd.read_csv(opt_uploaded)
        required_cols = {"date", "expense", "category", "is_forecast"}
        if not required_cols.issubset(df_in.columns):
            st.error(f"CSV must contain columns: {', '.join(sorted(required_cols))}")
            st.stop()

        df_in = df_in.copy()
        df_in["date"] = pd.to_datetime(df_in["date"])
        df_in["expense"] = df_in["expense"].astype(float)
        df_in["category"] = df_in["category"].astype(str)
        df_in["is_forecast"] = df_in["is_forecast"].astype(bool)
        opt_df = df_in.sort_values("date").reset_index(drop=True)

    # Top: show data + simple summary
    st.subheader("DataFrame")
    st.dataframe(opt_df.set_index("date"))

    with st.expander("Quick stats"):
        total_hist = opt_df.loc[~opt_df["is_forecast"], "expense"].sum()
        total_fc = opt_df.loc[opt_df["is_forecast"], "expense"].sum()
        st.write(f"Historical total: {total_hist:,.2f}")
        st.write(f"Forecasted total (marked as forecast): {total_fc:,.2f}")

    # Q&A interface
    st.subheader("Ask an optimization question")

    question = st.text_area(
        "Question",
        placeholder="Example: How can I reduce my food expenses next month?",
        height=80,
        key="opt_question",
    )

    if st.button("Generate answer", key="opt_answer_btn"):
        with st.spinner("Analyzing your expenses and forecast..."):
            answer = optimize_answer(opt_df, question)
        st.text_area("Answer", value=answer, height=160, key="opt_answer")
