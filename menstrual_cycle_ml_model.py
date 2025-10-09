import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib

class MenstrualCyclePredictionModel:
    def __init__(self):
        self.model = LinearRegression()  # Use only the best performing model
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
    def load_and_prepare_data(self, csv_file='Realdataboth.csv'):
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded data: {len(df)} samples")
            
            age_mapping = {'25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42}
            df['age_numeric'] = df['age_group'].map(age_mapping)
            
            df['phase_encoded'] = self.label_encoder.fit_transform(df['phase'])
            
            df['lh_to_prog_ratio'] = df['lh_level_miu_l'] / (df['progesterone_ng_ml'] + 0.1)
            df['hormone_interaction'] = df['lh_level_miu_l'] * df['progesterone_ng_ml']
            df['cycle_progress'] = df['cycle_day'] / 28
            
            feature_columns = [
                'age_numeric',
                'cycle_day', 
                'lh_level_miu_l',
                'progesterone_ng_ml',
                'phase_encoded',
                'lh_to_prog_ratio',
                'hormone_interaction',
                'cycle_progress'
            ]
            
            target = 'days_to_next_cycle'
            
            df_clean = df[feature_columns + [target] + ['age_group']].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target]
            age_groups = df_clean['age_group']
            
            valid_mask = (y >= 0) & (y <= 35) & (df_clean['lh_level_miu_l'] > 0) & (df_clean['progesterone_ng_ml'] > 0)
            X = X[valid_mask]
            y = y[valid_mask]
            age_groups = age_groups[valid_mask]
            
            self.feature_names = feature_columns
            
            print(f"Prepared data: {len(X)} samples with {len(feature_columns)} features")
            print(f"Age group distribution:")
            print(age_groups.value_counts())
            
            return X, y, age_groups
            
        except FileNotFoundError:
            print("CSV file not found.")
            return None, None, None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def stratified_train_test_split(self, X, y, age_groups, test_size=0.05, random_state=42):
        """
        Perform stratified sampling by age group - 80/20 split within each age group
        """
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        
        print(f"\nPerforming stratified split by age group ({(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test):")
        print("-" * 60)
        
        for age_group in age_groups.unique():
            # Get indices for this age group
            age_mask = age_groups == age_group
            X_age = X[age_mask]
            y_age = y[age_mask]
            
            if len(X_age) < 2:
                print(f"Age group {age_group}: {len(X_age)} samples - skipping (too few samples)")
                continue
                
            # Split this age group 80/20
            X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(
                X_age, y_age, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=True
            )
            
            X_train_list.append(X_train_age)
            X_test_list.append(X_test_age)
            y_train_list.append(y_train_age)
            y_test_list.append(y_test_age)
            
            print(f"Age group {age_group}: {len(X_age)} total → {len(X_train_age)} train, {len(X_test_age)} test")
        
        # Combine all age groups
        X_train = pd.concat(X_train_list, ignore_index=True)
        X_test = pd.concat(X_test_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        y_test = pd.concat(y_test_list, ignore_index=True)
        
        print(f"\nFinal split: {len(X_train)} train samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def calculate_accuracy_metrics(self, y_true, y_pred, model_name="Linear Regression"):
        """
        Calculate comprehensive accuracy metrics and explain day-range patterns
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage of predictions within different day ranges
        errors = np.abs(y_true - y_pred)
        within_1_day = np.mean(errors <= 1.0) * 100
        within_2_days = np.mean(errors <= 2.0) * 100
        within_3_days = np.mean(errors <= 3.0) * 100
        within_5_days = np.mean(errors <= 5.0) * 100
        
        print(f"\n{model_name} Performance Metrics:")
        print("=" * 50)
        print(f"Mean Absolute Error (MAE): {mae:.2f} days")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} days")
        print(f"R² Score: {r2:.3f}")
        print(f"\nAccuracy by Day Range:")
        print(f"  Within 1 day:  {within_1_day:.1f}%")
        print(f"  Within 2 days: {within_2_days:.1f}%")
        print(f"  Within 3 days: {within_3_days:.1f}%")
        print(f"  Within 5 days: {within_5_days:.1f}%")
        
        # Explain why accuracy changes by day range
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'within_1_day': within_1_day,
            'within_2_days': within_2_days,
            'within_3_days': within_3_days,
            'within_5_days': within_5_days
        }
    
    
    
    def train_model(self, X, y, age_groups):
        # Perform stratified split by age group
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(X, y, age_groups)
        
        # Scale features for Linear Regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nTraining Linear Regression model with stratified age group sampling")
        print("=" * 70)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate comprehensive metrics with explanation
        metrics = self.calculate_accuracy_metrics(y_test, y_pred)
        
        self.is_trained = True
        
        # Show feature coefficients (Linear Regression specific)
        print(f"\nLinear Regression Coefficients:")
        print("-" * 40)
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        for _, row in coef_df.iterrows():
            impact = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"{row['feature']:20}: {row['coefficient']:6.3f} ({impact} days to next period)")
        
        return X_test, y_test, metrics
    
    def predict_next_period(self, age, lh_level, progesterone_level, current_cycle_day=None):
        if not self.is_trained:
            print("Model not trained.")
            return None
        
        if current_cycle_day is None:
            if progesterone_level > 5.0:
                current_cycle_day = np.random.randint(16, 25)
            elif lh_level > 10.0:
                current_cycle_day = np.random.randint(12, 16)
            else:
                current_cycle_day = np.random.randint(6, 15)
        
        if progesterone_level > 4.0:
            phase = 'luteal'
            phase_encoded = self.label_encoder.transform(['luteal'])[0]
        else:
            phase = 'follicular'  
            phase_encoded = self.label_encoder.transform(['follicular'])[0]
        
        lh_to_prog_ratio = lh_level / (progesterone_level + 0.1)
        hormone_interaction = lh_level * progesterone_level
        cycle_progress = current_cycle_day / 28
        
        features = np.array([[
            age,
            current_cycle_day,
            lh_level,
            progesterone_level,
            phase_encoded,
            lh_to_prog_ratio,
            hormone_interaction,
            cycle_progress
        ]])
        
        # Scale features for Linear Regression
        features_scaled = self.scaler.transform(features)
        
        days_until_period = self.model.predict(features_scaled)[0]
        days_until_period = max(0, min(35, days_until_period))
        
        predicted_date = datetime.now() + timedelta(days=int(days_until_period))
        
        # Improved confidence based on typical prediction ranges
        confidence_score = max(0, 100 - abs(28 - days_until_period) * 3)
        
        results = {
            'days_until_next_period': round(days_until_period, 1),
            'predicted_date': predicted_date.strftime('%Y-%m-%d'),
            'current_phase': phase,
            'confidence_score': round(confidence_score, 1),
            'input_summary': {
                'age': age,
                'lh_level': lh_level,
                'progesterone_level': progesterone_level,
                'estimated_cycle_day': current_cycle_day
            }
        }
        
        return results
    
    def save_model(self, filename='menstrual_cycle_model.pkl'):
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filename)
            print(f"Model saved as '{filename}'")
        else:
            print("No trained model to save")
    
    def load_model(self, filename='menstrual_cycle_model.pkl'):
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            print(f"Model loaded from '{filename}'")
        except FileNotFoundError:
            print(f"Model file '{filename}' not found")


if __name__ == "__main__":
    print("MENSTRUAL CYCLE PREDICTION - LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    # Initialize the model
    predictor = MenstrualCyclePredictionModel()
    
    # Load and prepare data
    X, y, age_groups = predictor.load_and_prepare_data()
    
    if X is not None:
        # Train the model with stratified age group sampling
        X_test, y_test, metrics = predictor.train_model(X, y, age_groups)
        
        # Save the model
        predictor.save_model()
        
        print("\n" + "=" * 70)
        print("INTERACTIVE PREDICTION MODE")
        print("=" * 70)
        
        # Interactive prediction loop
        while True:
            try:
                print("\nEnter your hormone data (or type 'q' to quit):")
                age = input("Age (years): ")
                if age.lower() == "q":
                    break
                age = int(age)
                
                lh_level = float(input("LH level (mIU/L): "))
                progesterone_level = float(input("Progesterone level (ng/mL): "))
                
                cycle_day_input = input("Current cycle day (1–35, leave blank to auto-estimate): ")
                cycle_day = int(cycle_day_input) if cycle_day_input.strip() else None
                
                result = predictor.predict_next_period(
                    age=age,
                    lh_level=lh_level,
                    progesterone_level=progesterone_level,
                    current_cycle_day=cycle_day
                )
                
                if result:
                    print("\nPREDICTION RESULTS:")
                    print(f"Next period in: {result['days_until_next_period']} days")
                    print(f"Predicted date: {result['predicted_date']}")
                    print(f"Confidence score: {result['confidence_score']}%")
                    print(f"Current phase: {result['current_phase']}")
                    
            except ValueError:
                print("Please enter valid numbers for age and hormone levels.")
            except Exception as e:
                print(f"Input error: {e}. Please try again.")
    
    else:
        print("Could not load data. Please check your CSV file.")