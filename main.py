import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baca data
data = pd.read_csv('data_emisi_terbaru.csv')

# Filter data untuk Indonesia
indonesia_data = data[data['country'] == 'Indonesia']

# Hitung total emisi (CO2 + land use change CO2)
indonesia_data['total_emissions'] = indonesia_data['co2'] + indonesia_data['land_use_change_co2']

# Pilih variabel yang tersedia dan relevan
variables = [
    'year', 'co2', 'land_use_change_co2', 'total_emissions',
    'gdp', 'population', 'primary_energy_consumption',
    'share_of_temperature_change_from_ghg', 'energy_per_capita',
    'energy_per_gdp', 'co2_per_gdp', 'co2_per_capita'
]

# Buat dataframe baru dengan variabel terpilih
analysis_data = indonesia_data[variables].set_index('year')

# Analisis tren emisi
plt.figure(figsize=(12, 6))
plt.plot(analysis_data.index, analysis_data['total_emissions'], label='Total Emissions')
plt.plot(analysis_data.index, analysis_data['co2'], label='CO2 Emissions')
plt.plot(analysis_data.index, analysis_data['land_use_change_co2'], label='Land Use Change CO2')
plt.title('Trend Emisi CO2 Indonesia (2009-2022)')
plt.xlabel('Tahun')
plt.ylabel('Emisi (MtCO2)')
plt.legend()
plt.grid(True)
plt.show()

# Analisis sektoral
sectors = ['coal_co2', 'gas_co2', 'oil_co2', 'cement_co2']
sector_data = indonesia_data[['year'] + sectors].set_index('year')
sector_data.plot(kind='area', stacked=True, figsize=(12, 6))
plt.title('Emisi CO2 Sektoral Indonesia (2009-2022)')
plt.xlabel('Tahun')
plt.ylabel('Emisi (MtCO2)')
plt.legend(title='Sektor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Analisis intensitas emisi dan konsumsi energi
plt.figure(figsize=(12, 6))
plt.plot(analysis_data.index, analysis_data['co2_per_gdp'], label='CO2 per GDP')
plt.plot(analysis_data.index, analysis_data['co2_per_capita'], label='CO2 per Capita')
plt.plot(analysis_data.index, analysis_data['energy_per_gdp'], label='Energy per GDP')
plt.plot(analysis_data.index, analysis_data['energy_per_capita'], label='Energy per Capita')
plt.title('Intensitas Emisi CO2 dan Konsumsi Energi Indonesia (2009-2022)')
plt.xlabel('Tahun')
plt.ylabel('Intensitas')
plt.legend()
plt.grid(True)
plt.show()

# Statistik deskriptif
print(analysis_data.describe())

# Korelasi
correlation = analysis_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Korelasi antar Variabel')
plt.tight_layout()
plt.show()

# Analisis pertumbuhan
growth_data = analysis_data.pct_change().mean() * 100
growth_data = growth_data.sort_values(ascending=False)a
plt.figure(figsize=(12, 6))
growth_data.plot(kind='bar')
plt.title('Rata-rata Pertumbuhan Tahunan Variabel (2009-2022)')
plt.xlabel('Variabel')
plt.ylabel('Pertumbuhan Rata-rata (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()