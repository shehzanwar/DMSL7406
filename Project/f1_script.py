import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# =========================================================================
# Step 1: Load All CSV Files
# =========================================================================
file_paths = glob.glob('f1datasets/*.csv')

f1_dfs = {} # Dictionary to hold DataFrames

print(f"Loading {len(file_paths)} CSV files...") 
for file_path in file_paths:
    basename = os.path.basename(file_path)
    df_name = basename.split('.')[0] # Get the file name without extension (results.csv runs into issues with 'results' keyword)
    try:
        f1_dfs[df_name] = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")


# =========================================================================
# Step 2: Cleaning and Preprocessing the Data
# =========================================================================
for df_name in f1_dfs:
    f1_dfs[df_name].replace(r'\\N', np.nan, inplace=True)

def time_to_ms(time_str): # Convert time strings like '1:23.456' to milliseconds
    if pd.isna(time_str):
        return np.nan
    
    parts = str(time_str).split(':')

    try: # Handle different time formats
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return int((minutes * 60 + seconds) * 1000)
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return int((hours * 3600 + minutes * 60 + seconds) * 1000)
        else:
            return np.nan
    except ValueError: # In case of conversion error
        return np.nan
print("Cleaning data...\nConverting time strings to milliseconds...")

# Data cleaning for specific DataFrames
# Results
df_results = f1_dfs['results'].copy()
df_results['fastestLapTime_ms'] = df_results['fastestLapTime'].apply(time_to_ms)
df_results['positionOrder'] = pd.to_numeric(df_results['positionOrder'], errors='coerce')
df_results['podium'] = df_results['positionOrder'].apply(lambda x: 1 if x <= 3 else 0)
df_results['grid'] = pd.to_numeric(df_results['grid'], errors='coerce')

# Qualifying
df_quali = f1_dfs['qualifying'].copy()
df_quali['q1_ms'] = df_quali['q1'].apply(time_to_ms)
df_quali['q2_ms'] = df_quali['q2'].apply(time_to_ms)
df_quali['q3_ms'] = df_quali['q3'].apply(time_to_ms)
df_quali_short = df_quali[['raceId', 'driverId', 'constructorId', 'position', 'q1_ms', 'q2_ms', 'q3_ms']]
df_quali_short = df_quali_short.rename(columns={'position': 'qualifying_position'})

# Races
df_races = f1_dfs['races'].copy()
df_races['date'] = pd.to_datetime(df_races['date'])
df_races_short = df_races[['raceId', 'year', 'round', 'circuitId', 'date']]

# Drivers
df_drivers = f1_dfs['drivers'].copy()
df_drivers['dob'] = pd.to_datetime(df_drivers['dob'])
df_drivers_short = df_drivers[['driverId', 'driverRef', 'nationality', 'dob']]

# Constructors
df_constructors = f1_dfs['constructors'].copy()
df_constructors_slim = df_constructors[['constructorId', 'name', 'nationality']]
df_constructors_slim = df_constructors_slim.rename(columns={'name': 'constructor_name'})


# =========================================================================
# Step 3: Feature Engineering
# =========================================================================
print("Engineering features from 'lap_times'...")
df_lap_times = f1_dfs['lap_times'].copy()
df_lap_times['milliseconds'] = pd.to_numeric(df_lap_times['milliseconds'], errors='coerce')

# Computing average and std of lap times
lap_stats = df_lap_times.groupby(['raceId', 'driverId'])['milliseconds'].agg(['mean', 'std']).reset_index()
lap_stats = lap_stats.rename(columns={'mean': 'avg_lap_time_ms', 'std': 'std_lap_time_ms'})

# Merging all relevant DataFrames
print("Merging DataFrames...")
master_f1 = pd.merge(df_results, df_races_short, on='raceId', how='left')
master_f1 = pd.merge(master_f1, df_quali_short, on=['raceId', 'driverId', 'constructorId'], how='left')
master_f1 = pd.merge(master_f1, df_drivers_short, on='driverId', how='left')
master_f1 = pd.merge(master_f1, df_constructors_slim, on='constructorId', how='left')
master_f1 = pd.merge(master_f1, lap_stats, on=['raceId', 'driverId'], how='left')

# Computing drivers' age at the time of race
master_f1['driver_age_at_race'] = (master_f1['date'] - master_f1['dob']).dt.days / 365.25
print("Data preparation complete. Master DataFrame ready for analysis.")


# =========================================================================
# Step 5: Creating Final Analysis DataFrame
# =========================================================================
print("Creating final DataFrame for modeling...")

key_cols = [
    # Response Variables (Y)
    'fastestLapTime_ms', 'positionOrder', 'podium',
    # Predictor Variables (X)
    'year', 'circuitId', 'grid', 'qualifying_position', 'q1_ms', 'q2_ms', 'q3_ms',
    'avg_lap_time_ms', 'std_lap_time_ms', 'driver_age_at_race',
    # ID Columns
    'driverRef', 'constructor_name'
]

final_df = master_f1[key_cols].copy()
final_df.dropna(subset=['fastestLapTime_ms'], inplace=True) # Drop rows with missing target variable

print(f"Final DataFrame shape: {final_df.shape[0]} rows and {final_df.shape[1]} columns.")


# =========================================================================
# Step 6: Graphical Analysis
# =========================================================================
print("Generating output visualizations...")

# Data Sample for Table 1
ds1 = final_df.head()
print("\nData Sample for Table 1:")
print(ds1[['fastestLapTime_ms', 'positionOrder', 'podium', 'year', 'grid', 'qualifying_position', 'avg_lap_time_ms', 'driverRef']].to_markdown(index=False))

# Descriptive Statistics
desc_stats = final_df[['fastestLapTime_ms', 'positionOrder', 'grid', 'qualifying_position', 'avg_lap_time_ms', 'driver_age_at_race']].describe()
print("\nDescriptive Statistics:")
print(desc_stats.to_markdown(floatfmt=".2f"))

# Correlation Heatmap
print("Generating correlation heatmap...")
num_cols_for_corr = ['fastestLapTime_ms', 'positionOrder', 'grid', 'qualifying_position', 'avg_lap_time_ms', 'std_lap_time_ms', 'driver_age_at_race', 'year']
corr_matrix = final_df[num_cols_for_corr].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Key F1 Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('f1_correlation_heatmap.png')

print("\n'f1_correlation_heatmap.png' is saved.")

# =========================================================================
# Step 7: Pre-Modeling Data Preparation
# =========================================================================
print("Preparing data for modeling...")

# Defining Predictors (X) and Target (y)
X = final_df.drop(columns=['fastestLapTime_ms', 'positionOrder', 'podium', 'driverRef', 'constructor_name'])
y1 = final_df['fastestLapTime_ms'] # Q1
y2 = final_df['positionOrder'] # Q2
y3 = final_df['podium'] # Q3

print(f"X shape: {X.shape}, y1 shape: {y1.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns) # Convert back to DataFrame in order to keep column names
print("Missing data imputed using median.")

# Split into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    X_imputed, y1, y2, y3, test_size=0.3, random_state=7406
)
print(f"Data split: {len(X_train)} training samples and {len(X_test)} testing samples.")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete. Data is ready for modeling.")
print("\nSample of Scaled Training Data:")
print(X_train_scaled[:5])




