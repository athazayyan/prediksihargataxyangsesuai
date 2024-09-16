import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data for ETS prices in Indonesia and projections
current_price_indonesia = 0.61
increase_per_year = 1.40
years_indonesia = np.arange(2024, 2031)
predicted_prices_indonesia = current_price_indonesia + (years_indonesia - 2024) * increase_per_year

# Data for CO2 emissions and projected reduction
initial_emissions = 629.83  # CO2 emissions in 2023
target_reduction = 0.29     # 29% target reduction in emissions
years_emission = np.arange(2023, 2031)
predicted_emissions = initial_emissions * (1 - np.linspace(0, target_reduction, len(years_emission)))

# Function to display the ETS price projection graph
def plot_price_projection(selected_year):
    fig, ax = plt.subplots()
    ax.plot(years_indonesia, predicted_prices_indonesia, color='green', linestyle='-', marker='o', label='Projected ETS Price in Indonesia')
    ax.axvline(x=selected_year, color='red', linestyle='--', label=f'Slider Year: {selected_year}')
    ax.set_xlabel('Year')
    ax.set_ylabel('ETS Price ($)')
    ax.set_title('Projected ETS Price in Indonesia')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Function to display the CO2 emission reduction graph
def plot_emission_reduction():
    fig, ax = plt.subplots()
    ax.plot(years_emission, predicted_emissions, color='blue', linestyle='-', marker='o', label='Projected CO2 Emission Reduction')
    ax.axhline(y=predicted_emissions[-1], color='red', linestyle='--', label=f'2030 Emission Target: {predicted_emissions[-1]:.2f}')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (kilo-tons)')
    ax.set_title('Projected CO2 Emission Reduction in Indonesia')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Streamlit app
st.title("Projected ETS Prices and CO2 Emission Reduction in Indonesia")

# Slider to select the year
selected_year = st.slider('Select Year', min_value=2024, max_value=2030, value=2024)

# Display the projected ETS price for the selected year
price_for_year = current_price_indonesia + (selected_year - 2024) * increase_per_year
st.write(f"Projected ETS Price in Indonesia for {selected_year}: ${price_for_year:.2f}")

# Plot the ETS price projection graph
plot_price_projection(selected_year)

# Plot the CO2 emission reduction graph
plot_emission_reduction()
