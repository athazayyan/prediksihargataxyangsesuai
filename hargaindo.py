import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data harga ETS Indonesia dan proyeksi
current_price_indonesia = 0.61
increase_per_year = 1.40
years_indonesia = np.arange(2024, 2031)
predicted_prices_indonesia = current_price_indonesia + (years_indonesia - 2024) * increase_per_year

# Fungsi untuk menampilkan grafik
def plot_price_projection(selected_year):
    fig, ax = plt.subplots()
    ax.plot(years_indonesia, predicted_prices_indonesia, color='green', linestyle='-', marker='o', label='Proyeksi Harga ETS Indonesia')
    ax.axvline(x=selected_year, color='red', linestyle='--', label=f'Slider Tahun: {selected_year}')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Harga ETS ($)')
    ax.set_title('Proyeksi Harga ETS di Indonesia')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Streamlit app
st.title("Proyeksi Harga ETS Indonesia")

# Slider untuk memilih tahun
selected_year = st.slider('Pilih Tahun', min_value=2024, max_value=2030, value=2024)

# Menampilkan proyeksi harga ETS untuk tahun yang dipilih
price_for_year = current_price_indonesia + (selected_year - 2024) * increase_per_year
st.write(f"Harga ETS Indonesia pada tahun {selected_year}: ${price_for_year:.2f}")

# Plot grafik
plot_price_projection(selected_year)
