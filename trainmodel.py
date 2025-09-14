import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
        """Load and prepare the hormone data for ML training"""
        try:
            # Load the data
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded data: {len(df)} samples")
            
            # Convert age_group to numeric
            age_mapping = {'25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42}
            df['age_numeric'] = df['age_group'].map(age_mapping)
            
            # Encode phase to numeric
            df['phase_encoded'] = self.label_encoder.fit_transform(df['phase'])
            
            # Create additional features
            df['lh_to_prog_ratio'] = df['lh_level_miu_l'] / (df['progesterone_ng_ml'] + 0.1)
            df['hormone_interaction'] = df['lh_level_miu_l'] * df['progesterone_ng_ml']
            df['cycle_progress'] = df['cycle_day'] / 28  # Normalized cycle progress
            
            # Select features for the model
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
            
            # Target variable
            target = 'days_to_next_cycle'
            
            # Remove any rows with missing values
            df_clean = df[feature_columns + [target]].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target]
            
            # Filter out unrealistic values
            valid_mask = (y >= 0) & (y <= 35) & (df_clean['lh_level_miu_l'] > 0) & (df_clean['progesterone_ng_ml'] > 0)
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.feature_names = feature_columns
            
            print(f"✅ Prepared data: {len(X)} samples with {len(feature_columns)} features")
            print(f"✅ Target range: {y.min():.1f} to {y.max():.1f} days")
            
            return X, y, df_clean[valid_mask]
            
        except FileNotFoundError:
            print("❌ CSV file not found. Please make sure 'comprehensive_hormone_cycle_data.csv' exists.")
            return None, None, None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None
    
    def train_model(self, X, y):
        """Train multiple models and select the best one"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model = None
        best_name = None
        
        print("\n🔬 Training and evaluating models...")
        print("-" * 50)
        
        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name}:")
            print(f"  MAE: {mae:.2f} days")
            print(f"  RMSE: {np.sqrt(mse):.2f} days")
            print(f"  R²: {r2:.3f}")
            print()
            
            if mae < best_score:
                best_score = mae
                best_model = model
                best_name = name
        
        # Store the best model
        self.model = best_model
        self.is_trained = True
        
        print(f"🏆 Best model: {best_name} (MAE: {best_score:.2f} days)")
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Feature Importance:")
            print(importance_df.to_string(index=False))
        
        return X_test, y_test
    
    def predict_next_period(self, age, lh_level, progesterone_level, current_cycle_day=None):
        """
        Predict when the next period will occur
        
        Parameters:
        - age: Age in years (18-45)
        - lh_level: LH level in mIU/L
        - progesterone_level: Progesterone level in ng/mL
        - current_cycle_day: Current day of cycle (optional, will estimate if not provided)
        """
        
        if not self.is_trained:
            print("❌ Model not trained. Please train the model first.")
            return None
        
        # Estimate cycle day if not provided
        if current_cycle_day is None:
            if progesterone_level > 5.0:
                current_cycle_day = np.random.randint(16, 25)  # Likely luteal phase
            elif lh_level > 10.0:
                current_cycle_day = np.random.randint(12, 16)  # Likely ovulation
            else:
                current_cycle_day = np.random.randint(6, 15)   # Likely follicular
            print(f"📅 Estimated current cycle day: {current_cycle_day}")
        
        # Determine phase based on hormone levels
        if progesterone_level > 4.0:
            phase = 'luteal'
            phase_encoded = self.label_encoder.transform(['luteal'])[0]
        else:
            phase = 'follicular'  
            phase_encoded = self.label_encoder.transform(['follicular'])[0]
        
        # Calculate additional features
        lh_to_prog_ratio = lh_level / (progesterone_level + 0.1)
        hormone_interaction = lh_level * progesterone_level
        cycle_progress = current_cycle_day / 28
        
        # Create feature array
        features = np.array([[
            age,                    # age_numeric
            current_cycle_day,      # cycle_day
            lh_level,              # lh_level_miu_l
            progesterone_level,    # progesterone_ng_ml
            phase_encoded,         # phase_encoded
            lh_to_prog_ratio,      # lh_to_prog_ratio
            hormone_interaction,   # hormone_interaction
            cycle_progress         # cycle_progress
        ]])
        
        # Scale features if using Linear Regression
        if isinstance(self.model, LinearRegression):
            features = self.scaler.transform(features)
        
        # Make prediction
        days_until_period = self.model.predict(features)[0]
        days_until_period = max(0, min(35, days_until_period))  # Reasonable bounds
        
        # Calculate predicted date
        predicted_date = datetime.now() + timedelta(days=int(days_until_period))
        
        # Results
        results = {
            'days_until_next_period': round(days_until_period, 1),
            'predicted_date': predicted_date.strftime('%Y-%m-%d'),
            'current_phase': phase,
            'confidence': 'high' if 1 <= days_until_period <= 30 else 'low',
            'input_summary': {
                'age': age,
                'lh_level': lh_level,
                'progesterone_level': progesterone_level,
                'estimated_cycle_day': current_cycle_day
            }
        }
        
        return results
    
    def save_model(self, filename='menstrual_cycle_model.pkl'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filename)
            print(f"✅ Model saved as '{filename}'")
        else:
            print("❌ No trained model to save")
    
    def load_model(self, filename='menstrual_cycle_model.pkl'):
        """Load a previously trained model"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            print(f"✅ Model loaded from '{filename}'")
        except FileNotFoundError:
            print(f"❌ Model file '{filename}' not found")

# Main execution
if __name__ == "__main__":
    print("🩸 MENSTRUAL CYCLE PREDICTION MODEL")
    print("=" * 50)
    
    # Initialize the model
    predictor = MenstrualCyclePredictionModel()
    
    # Load and prepare data
    X, y, df = predictor.load_and_prepare_data()
    
    if X is not None:
        # Train the model
        X_test, y_test = predictor.train_model(X, y)
        
        # Save the model
        predictor.save_model()
        
        print("\n" + "=" * 50)
        print("🎯 MODEL READY FOR PREDICTIONS!")
        print("=" * 50)
        
        # Example predictions
        print("\n📋 EXAMPLE PREDICTIONS:")
        print("-" * 30)
        
        # Example 1: Young woman, follicular phase
        result1 = predictor.predict_next_period(
            age=28,
            lh_level=5.2,
            progesterone_level=2.1,
            current_cycle_day=12
        )
        
        if result1:
            print(f"\n👤 Example 1 - Age 28, Follicular Phase:")
            print(f"   LH: {result1['input_summary']['lh_level']} mIU/L")
            print(f"   Progesterone: {result1['input_summary']['progesterone_level']} ng/mL")
            print(f"   ➡️ Next period in: {result1['days_until_next_period']} days")
            print(f"   📅 Predicted date: {result1['predicted_date']}")
            print(f"   🔍 Confidence: {result1['confidence']}")
        
        # Example 2: Older woman, luteal phase
        result2 = predictor.predict_next_period(
            age=35,
            lh_level=4.1,
            progesterone_level=8.7,
            current_cycle_day=20
        )
        
        if result2:
            print(f"\n👤 Example 2 - Age 35, Luteal Phase:")
            print(f"   LH: {result2['input_summary']['lh_level']} mIU/L")
            print(f"   Progesterone: {result2['input_summary']['progesterone_level']} ng/mL")
            print(f"   ➡️ Next period in: {result2['days_until_next_period']} days")
            print(f"   📅 Predicted date: {result2['predicted_date']}")
            print(f"   🔍 Confidence: {result2['confidence']}")
        
        print("\n" + "=" * 50)
        print("✨ TO USE THIS MODEL:")
        print("1. Run this script to train the model")
        print("2. Use predict_next_period(age, lh_level, progesterone_level)")
        print("3. Input your current hormone test results")
        print("4. Get prediction for your next period!")
        print("=" * 50)
    
    else:
        print("❌ Could not load data. Please check your CSV file.")

# Additional utility function for easy predictions
def quick_prediction(age, lh_level, progesterone_level, cycle_day=None):
    """Quick prediction function - loads model and makes prediction"""
    predictor = MenstrualCyclePredictionModel()
    try:
        predictor.load_model()
        return predictor.predict_next_period(age, lh_level, progesterone_level, cycle_day)
    except:
        print("❌ Please train the model first by running the main script")
        return None

print("\n💡 USAGE EXAMPLES:")
print("# After training, you can use:")
print("result = quick_prediction(age=30, lh_level=6.5, progesterone_level=3.2)")
print("print(f'Next period in {result[\"days_until_next_period\"]} days')")