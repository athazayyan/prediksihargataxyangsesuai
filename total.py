import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_data():
    # Load original CSV data
    df = pd.read_csv('data_emisi_terbaru.csv')
    df = df[df['country'] == 'Indonesia'].set_index('year')
    
    # Add new data
    df['carbon_price'] = pd.Series({2022: 30000}, index=df.index)  # Assuming 30000 IDR/tCO2 for 2022
    
    # Add industrial production index
    prod_index = pd.Series({
        2010: 4, 2011: 4, 2012: 5, 2013: 4, 2014: 4, 2015: 4, 2016: 4, 2017: 4, 
        2018: 4, 2019: 3, 2020: -10, 2021: 16, 2022: 4
    })
    df['industrial_production_index'] = prod_index
    
    # Add renewable energy percentage
    renewable_energy = pd.Series({
        2015: 4.90, 2016: 6.27, 2017: 6.66, 2018: 8.60, 2019: 9.19, 
        2020: 11.27, 2021: 12.16, 2022: 12.30
    })
    df['renewable_energy_percentage'] = renewable_energy
    
    # Add crude oil price
    oil_price = pd.Series({2017: 51.31, 2018: 68.16, 2019: 65.50, 2020: 44.52, 2021: 70.65, 2022: 114.27})
    df['crude_oil_price'] = oil_price
    
    # Add energy sector investment (converting to billion IDR assuming 1 USD = 14000 IDR)
    investment = pd.Series({
        2010: 14 * 14000, 2012: 33.77 * 14000, 2013: 376000, 2014: 34 * 14000, 
        2015: 29.4, 2016: 27 * 14000, 2017: 27.5 * 14000, 2018: 31.2 * 14000, 
        2019: 30.6 * 14000, 2020: 26.3 * 14000, 2021: 27.5 * 14000, 2022: 27 * 14000
    })
    df['energy_investment'] = investment
    
    # Add environmental policy index
    policy_index = pd.Series({2018: 65.14, 2019: 66.55, 2020: 70.27, 2021: 71.45, 2022: 72.42})
    df['environmental_policy_index'] = policy_index
    
    # Add forest area data
    forest_data = pd.DataFrame({
        'year': range(2014, 2023),
        'forest_area': [120770.3, 120562.4, 120423.8, 120390.1, 120385.7, 120281.6, 120261.6, 118365.5, 118194.7],
        'non_forest_area': [66981.6, 67189.5, 67328.0, 67361.7, 67366.2, 67470.3, 67490.2, 69313.7, 69393.0],
        'total_area': [187751.9, 187751.9, 187751.9, 187751.9, 187751.9, 187751.9, 187751.9, 187679.0, 187587.6]
    }).set_index('year')
    df = df.join(forest_data)
    
    return df

df = load_data()

# Streamlit app
st.title('Analisis Emisi GRK dan Harga Karbon Indonesia')

# Historical trends
st.header('Tren Historis')
variables = st.multiselect('Pilih variabel untuk ditampilkan:', df.columns.tolist(), default=['co2', 'gdp'])
fig, ax = plt.subplots(figsize=(12, 6))
for var in variables:
    ax.plot(df.index, df[var], label=var)
ax.set_xlabel('Tahun')
ax.set_ylabel('Nilai')
ax.legend()
st.pyplot(fig)

# GHG Emissions Prediction
st.header('Prediksi Emisi GRK')
features = ['gdp', 'population', 'primary_energy_consumption', 'industrial_production_index', 'renewable_energy_percentage', 'crude_oil_price', 'energy_investment']
available_features = [f for f in features if f in df.columns]

X = df[available_features].dropna()
y = df['co2'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

future_years = st.slider('Pilih tahun untuk prediksi:', 2023, 2030, 2025)
future_X = X.iloc[-1:].copy()
predictions = []

for year in range(df.index.max() + 1, future_years + 1):
    future_X.index = [year]
    pred = model.predict(future_X)
    predictions.append((year, pred[0]))
    future_X['gdp'] *= 1.05  # Assuming 5% annual GDP growth
    future_X['population'] *= 1.01  # Assuming 1% annual population growth
    # Add assumptions for other features if necessary

predictions_df = pd.DataFrame(predictions, columns=['year', 'predicted_co2']).set_index('year')
st.write(predictions_df)

# Carbon price recommendation
st.header('Rekomendasi Harga Karbon')
current_price = df['carbon_price'].iloc[-1]
emissions_change = (predictions_df['predicted_co2'].iloc[-1] - df['co2'].iloc[-1]) / df['co2'].iloc[-1]

if emissions_change > 0:
    recommended_price = current_price * (1 + emissions_change)
else:
    recommended_price = current_price

st.write(f'Harga karbon saat ini: Rp {current_price:,.0f} per tCO2')
st.write(f'Rekomendasi harga karbon untuk {future_years}: Rp {recommended_price:,.0f} per tCO2')

st.write('Catatan: Rekomendasi ini berdasarkan perubahan prediksi emisi dan harus dipertimbangkan bersama dengan faktor-faktor ekonomi dan kebijakan lainnya.')

# Correlation analysis
st.header('Analisis Korelasi')
corr_matrix = df[available_features + ['co2']].corr()
fig, ax = plt.subplots(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Matriks Korelasi')
st.pyplot(fig)

# Recommendations
st.header('Rekomendasi Kebijakan')
st.write("""
Berdasarkan analisis di atas, berikut beberapa rekomendasi kebijakan:

1. Pertimbangkan untuk meningkatkan harga karbon secara bertahap sesuai dengan proyeksi emisi.
2. Fokus pada peningkatan persentase energi terbarukan dalam bauran energi untuk mengurangi emisi.
3. Dorong investasi di sektor energi bersih untuk mendukung transisi energi.
4. Implementasikan kebijakan untuk meningkatkan efisiensi energi di sektor industri.
5. Pertahankan dan tingkatkan luas tutupan hutan untuk menjaga penyerapan karbon alami.
6. Lakukan pemantauan dan evaluasi berkala terhadap efektivitas kebijakan harga karbon.
""")

st.write('Catatan: Aplikasi ini menggunakan data historis dan proyeksi sederhana. Untuk pengambilan keputusan yang lebih akurat, diperlukan analisis yang lebih mendalam dan pertimbangan faktor-faktor tambahan.')