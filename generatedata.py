import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hormone_data(n_women=1000, cycles_per_woman=6):
    """
    Generate synthetic hormone data based on established medical reference ranges
    """
    data = []
    
    for woman_id in range(1, n_women + 1):
        # Generate woman-specific characteristics
        age = np.random.normal(28, 6)  # Age between 18-40
        age = max(18, min(40, age))
        
        # Typical cycle length varies between women
        base_cycle_length = np.random.normal(28, 3)
        base_cycle_length = max(21, min(35, base_cycle_length))
        
        for cycle in range(cycles_per_woman):
            # Cycle length can vary slightly for each cycle
            cycle_length = max(21, min(35, np.random.normal(base_cycle_length, 1.5)))
            cycle_length = int(cycle_length)
            
            # Generate start date
            start_date = datetime(2023, 1, 1) + timedelta(days=cycle * 30 + np.random.randint(-5, 5))
            
            for day in range(1, cycle_length + 1):
                current_date = start_date + timedelta(days=day-1)
                
                # Determine cycle phase
                if day <= 5:
                    phase = "menstrual"
                elif day <= cycle_length // 2 - 2:
                    phase = "follicular"
                elif day <= cycle_length // 2 + 2:
                    phase = "ovulatory"
                else:
                    phase = "luteal"
                
                # Generate hormone levels based on phase and established ranges
                # Reference ranges from medical literature
                
                if phase == "menstrual":
                    # Menstrual phase (days 1-5)
                    estradiol = np.random.normal(50, 20)  # pg/mL
                    progesterone = np.random.normal(0.5, 0.3)  # ng/mL
                    fsh = np.random.normal(6, 2)  # mIU/mL
                    lh = np.random.normal(5, 2)  # mIU/mL
                    
                elif phase == "follicular":
                    # Follicular phase (days 6-12)
                    estradiol = np.random.normal(80 + (day-6)*10, 25)  # Rising
                    progesterone = np.random.normal(0.8, 0.4)  # Low
                    fsh = np.random.normal(7, 2.5)
                    lh = np.random.normal(6, 2.5)
                    
                elif phase == "ovulatory":
                    # Ovulatory phase (days 13-15)
                    peak_day = cycle_length // 2
                    if day == peak_day:
                        # LH surge
                        lh = np.random.normal(25, 8)  # Peak
                        estradiol = np.random.normal(200, 50)  # Peak
                    else:
                        lh = np.random.normal(15, 5)
                        estradiol = np.random.normal(150, 40)
                    
                    progesterone = np.random.normal(1.5, 0.8)
                    fsh = np.random.normal(8, 3)
                    
                else:  # luteal
                    # Luteal phase (days 16-28)
                    days_since_ovulation = day - (cycle_length // 2)
                    estradiol = np.random.normal(120 - days_since_ovulation*5, 30)
                    progesterone = np.random.normal(12 + days_since_ovulation*2, 4)  # High
                    fsh = np.random.normal(4, 1.5)  # Suppressed
                    lh = np.random.normal(4, 1.5)  # Suppressed
                
                # Ensure realistic bounds
                estradiol = max(10, estradiol)
                progesterone = max(0.1, progesterone)
                fsh = max(0.5, min(20, fsh))
                lh = max(0.5, min(40, lh))
                
                # Calculate days until next cycle
                days_to_next_cycle = cycle_length - day
                
                # Add some individual variation
                age_factor = 1 + (age - 28) * 0.01  # Slight age effect
                estradiol *= age_factor
                progesterone *= age_factor
                
                data.append({
                    'woman_id': woman_id,
                    'age': round(age, 1),
                    'cycle_number': cycle + 1,
                    'cycle_day': day,
                    'cycle_length': cycle_length,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'phase': phase,
                    'estradiol_pg_ml': round(estradiol, 1),
                    'progesterone_ng_ml': round(progesterone, 2),
                    'fsh_miu_ml': round(fsh, 1),
                    'lh_miu_ml': round(lh, 1),
                    'days_to_next_cycle': days_to_next_cycle,
                    'bmi': round(np.random.normal(24, 4), 1),  # Additional feature
                    'stress_level': np.random.randint(1, 6),  # 1-5 scale
                })
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating synthetic hormone dataset...")
df = generate_hormone_data(n_women=500, cycles_per_woman=4)

# Add some derived features that might be useful for ML
df['estradiol_to_progesterone_ratio'] = df['estradiol_pg_ml'] / df['progesterone_ng_ml']
df['is_ovulatory_phase'] = (df['phase'] == 'ovulatory').astype(int)
df['cycle_progress'] = df['cycle_day'] / df['cycle_length']

print(f"Dataset created with {len(df)} records")
print(f"Number of unique women: {df['woman_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
print(df.head(10))
print("\nHormone level statistics by phase:")
print(df.groupby('phase')[['estradiol_pg_ml', 'progesterone_ng_ml', 'fsh_miu_ml', 'lh_miu_ml']].describe())

# Save to CSV
csv_filename = 'menstrual_cycle_hormone_data.csv'
df.to_csv(csv_filename, index=False)
print(f"\nDataset saved as '{csv_filename}'")

# Display sample predictions setup
print("\n" + "="*60)
print("SAMPLE ML MODEL SETUP")
print("="*60)
print("""
Target variable suggestions:
1. 'days_to_next_cycle' - Regression problem
2. 'phase' - Classification problem  
3. Binary classification: next cycle in <7 days, 7-14 days, >14 days

Key features for your model:
- estradiol_pg_ml, progesterone_ng_ml, fsh_miu_ml, lh_miu_ml
- age, cycle_day, cycle_progress
- estradiol_to_progesterone_ratio
- bmi, stress_level (additional factors)

Sample code to get started:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('menstrual_cycle_hormone_data.csv')

# Features for prediction
features = ['estradiol_pg_ml', 'progesterone_ng_ml', 'fsh_miu_ml', 'lh_miu_ml', 
           'age', 'cycle_progress', 'bmi', 'estradiol_to_progesterone_ratio']

X = df[features]
y = df['days_to_next_cycle']  # Predict days until next cycle

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f} days')
```
""")