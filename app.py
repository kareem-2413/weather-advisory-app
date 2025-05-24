
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# Load and preprocess data
def load_data():
    df = pd.read_csv("preprocessed_weather_data.csv", parse_dates=['date'])
    return df

def generate_forecast(df, mandal):
    data = df[df['mandal'] == mandal].copy()
    data.set_index('date', inplace=True)
    data = data.asfreq('W', method='pad')
    data['weekofyear'] = data.index.isocalendar().week
    for lag in range(1, 5):
        data[f'rainfall_lag_{lag}'] = data['rainfall'].shift(lag)
    data.dropna(inplace=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = data[[f'rainfall_lag_{i}' for i in range(1, 5)] + ['weekofyear']]
    y = data['rainfall']
    model.fit(X, y)

    recent = data.tail(4).copy()
    forecast = []
    for i in range(1, 53):
        features = {
            'weekofyear': i,
            'rainfall_lag_1': recent.iloc[-1]['rainfall'],
            'rainfall_lag_2': recent.iloc[-2]['rainfall'],
            'rainfall_lag_3': recent.iloc[-3]['rainfall'],
            'rainfall_lag_4': recent.iloc[-4]['rainfall']
        }
        pred = model.predict([list(features.values())])[0]
        forecast.append({'week': i, 'rainfall': pred})
        next_row = pd.DataFrame([{'rainfall': pred, **features}], index=[recent.index[-1] + pd.Timedelta(weeks=1)])
        recent = pd.concat([recent, next_row])

    forecast_df = pd.DataFrame(forecast)
    forecast_df['date'] = pd.date_range(start='2024-01-07', periods=52, freq='W')
    forecast_df['month'] = forecast_df['date'].dt.strftime('%b')
    return forecast_df

def generate_advice(row):
    advice = "Water Needed"
    if row['rainfall'] > 2:
        advice = "No Irrigation Needed"
    pesticide = "Low Risk"
    if row['rainfall'] > 5:
        pesticide = "High Risk: Spray Fungicide"
    elif row['rainfall'] > 2:
        pesticide = "Moderate Risk: Monitor"
    return pd.Series([advice, pesticide], index=['irrigation_advice', 'pesticide_advice'])

# Streamlit UI
df = load_data()
mandals = df['mandal'].unique().tolist()
st.sidebar.title("Papaya Farm Weather Advisory")
selected_mandals = st.sidebar.multiselect("Select Mandals", mandals, default=[mandals[0]])

if len(selected_mandals) == 1:
    forecast_df = generate_forecast(df, selected_mandals[0])
    forecast_df[['irrigation_advice', 'pesticide_advice']] = forecast_df.apply(generate_advice, axis=1)

    st.title(f"2024 Forecast & Advisory for {selected_mandals[0]}")
    fig, ax = plt.subplots()
    ax.plot(forecast_df['date'], forecast_df['rainfall'], marker='o')
    ax.set_title("Weekly Rainfall Prediction (2024)")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    st.subheader("Advisory Table")
    st.dataframe(forecast_df[['date', 'month', 'week', 'rainfall', 'irrigation_advice', 'pesticide_advice']])

else:
    st.title("Mandal Comparison Dashboard")
    all_forecasts = []
    for mandal in selected_mandals:
        forecast_df = generate_forecast(df, mandal)
        forecast_df['mandal'] = mandal
        all_forecasts.append(forecast_df)
    combined_df = pd.concat(all_forecasts)

    fig, ax = plt.subplots()
    for mandal in selected_mandals:
        subset = combined_df[combined_df['mandal'] == mandal]
        ax.plot(subset['date'], subset['rainfall'], label=mandal)
    ax.set_title("Rainfall Comparison for Selected Mandals")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecast Table by Mandal")
    st.dataframe(combined_df[['mandal', 'date', 'month', 'week', 'rainfall']])
