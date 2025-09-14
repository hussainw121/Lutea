import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib

class MenstrualCyclePredictionModel:
    def __init__(self):
        self.model = None
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
            
            df_clean = df[feature_columns + [target]].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target]
            
            valid_mask = (y >= 0) & (y <= 35) & (df_clean['lh_level_miu_l'] > 0) & (df_clean['progesterone_ng_ml'] > 0)
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.feature_names = feature_columns
            
            print(f"Prepared data: {len(X)} samples with {len(feature_columns)} features")
            return X, y, df_clean[valid_mask]
            
        except FileNotFoundError:
            print("CSV file not found.")
            return None, None, None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model = None
        best_name = None
        
        print("\nTraining and evaluating models")
        print("-" * 40)
        
        for name, model in models.items():
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name}: MAE={mae:.2f}, RMSE={np.sqrt(mse):.2f}, R²={r2:.3f}")
            
            if mae < best_score:
                best_score = mae
                best_model = model
                best_name = name
        
        self.model = best_model
        self.is_trained = True
        
        print(f"Best model: {best_name} (MAE: {best_score:.2f})")
        
        return X_test, y_test
    
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
        
        if isinstance(self.model, LinearRegression):
            features = self.scaler.transform(features)
        
        days_until_period = self.model.predict(features)[0]
        days_until_period = max(0, min(35, days_until_period))
        
        predicted_date = datetime.now() + timedelta(days=int(days_until_period))
        
        # Confidence score: inverse of distance from average cycle length (28 days)
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
    print("MENSTRUAL CYCLE PREDICTION MODEL")
    print("=" * 50)
    
    predictor = MenstrualCyclePredictionModel()
    X, y, df = predictor.load_and_prepare_data()
    
    if X is not None:
        X_test, y_test = predictor.train_model(X, y)
        predictor.save_model()
        
        print("\nMODEL READY FOR PREDICTIONS")
        print("=" * 50)
        
        while True:
            try:
                print("\nEnter your own data (or type 'q' to quit):")
                age = input("Age (years): ")
                if age.lower() == "q":
                    break
                age = int(age)
                
                lh_level = float(input("LH level (mIU/L): "))
                progesterone_level = float(input("Progesterone level (ng/mL): "))
                
                cycle_day_input = input("Current cycle day (1–28, leave blank to auto-estimate): ")
                cycle_day = int(cycle_day_input) if cycle_day_input.strip() else None
                
                result = predictor.predict_next_period(
                    age=age,
                    lh_level=lh_level,
                    progesterone_level=progesterone_level,
                    current_cycle_day=cycle_day
                )
                
                if result:
                    print("\nPrediction Results:")
                    print(f"Next period in: {result['days_until_next_period']} days")
                    print(f"Predicted date: {result['predicted_date']}")
                    print(f"Confidence score: {result['confidence_score']}%")
                    print(f"Current phase: {result['current_phase']}")
            except Exception as e:
                print(f"Input error: {e}. Please try again.")
