import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib

class UltimateMenstrualCycleModel:
    def __init__(self):
        # Using Gradient Boosting - captures non-linear patterns in hormones
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.prog_roc_data = None
        
    def calculate_rate_of_change(self, df):
        """Calculate rate of change for progesterone and LH"""
        df = df.sort_values(['subject_id', 'cycle_day'])
        
        df['prog_roc'] = df.groupby('subject_id')['progesterone_ng_ml'].diff()
        df['lh_roc'] = df.groupby('subject_id')['lh_level_miu_l'].diff()
        
        df['prog_roc'] = df['prog_roc'].fillna(0)
        df['lh_roc'] = df['lh_roc'].fillna(0)
        
        df['prog_declining'] = (df['prog_roc'] < -1.0).astype(int)
        df['prog_rapid_decline'] = (df['prog_roc'] < -2.0).astype(int)
        df['prog_stable_or_rising'] = (df['prog_roc'] >= 0).astype(int)
        
        df['lh_surging'] = (df['lh_roc'] > 2.0).astype(int)
        df['lh_post_surge_decline'] = (df['lh_roc'] < -2.0).astype(int)
        
        return df
    
    def load_and_prepare_data(self, csv_file='DataSet.csv'):
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded data: {len(df)} samples")
            
            age_mapping = {'25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42}
            df['age_numeric'] = df['age_group'].map(age_mapping)
            
            df['phase_encoded'] = self.label_encoder.fit_transform(df['phase'])
            
            print("\nCalculating rate of change features...")
            df = self.calculate_rate_of_change(df)
            
            self.prog_roc_data = df[['cycle_day', 'progesterone_ng_ml', 'prog_roc', 
                                     'days_to_next_cycle']].copy()
            
            # Progesterone absolute levels
            df['prog_premenstrual_low'] = (df['progesterone_ng_ml'] < 1.5).astype(int)
            df['prog_late_luteal'] = ((df['progesterone_ng_ml'] >= 1.5) & 
                                      (df['progesterone_ng_ml'] < 5.0)).astype(int)
            df['prog_mid_luteal'] = ((df['progesterone_ng_ml'] >= 10.0) & 
                                     (df['progesterone_ng_ml'] <= 25.0)).astype(int)
            df['prog_early_luteal'] = ((df['progesterone_ng_ml'] >= 5.0) & 
                                       (df['progesterone_ng_ml'] < 10.0)).astype(int)
            df['prog_follicular'] = (df['progesterone_ng_ml'] < 2.0).astype(int)
            df['prog_ovulation_confirmed'] = (df['progesterone_ng_ml'] >= 5.0).astype(int)
            
            # LH patterns
            df['lh_surge'] = (df['lh_level_miu_l'] > 10.0).astype(int)
            df['lh_elevated'] = ((df['lh_level_miu_l'] > 7.0) & 
                                 (df['lh_level_miu_l'] <= 10.0)).astype(int)
            df['lh_baseline'] = (df['lh_level_miu_l'] <= 7.0).astype(int)
            
            # Cycle phase
            df['late_cycle'] = (df['cycle_day'] > 20).astype(int)
            df['mid_cycle'] = ((df['cycle_day'] > 10) & (df['cycle_day'] <= 20)).astype(int)
            df['early_cycle'] = (df['cycle_day'] <= 10).astype(int)
            
            # Critical combinations
            df['period_warning_strong'] = ((df['late_cycle'] == 1) & 
                                          (df['prog_declining'] == 1) & 
                                          (df['progesterone_ng_ml'] < 5.0)).astype(int)
            
            df['period_warning_critical'] = ((df['late_cycle'] == 1) & 
                                            (df['prog_rapid_decline'] == 1) & 
                                            (df['progesterone_ng_ml'] < 3.0)).astype(int)
            
            df['post_ovulation_confirmed'] = ((df['mid_cycle'] == 1) & 
                                              (df['prog_mid_luteal'] == 1)).astype(int)
            
            # Hormone dynamics
            df['lh_to_prog_ratio'] = df['lh_level_miu_l'] / (df['progesterone_ng_ml'] + 0.1)
            df['prog_to_lh_ratio'] = df['progesterone_ng_ml'] / (df['lh_level_miu_l'] + 0.1)
            df['hormone_product'] = df['lh_level_miu_l'] * df['progesterone_ng_ml']
            
            df['prog_lh_roc_interaction'] = df['prog_roc'] * df['lh_roc']
            
            # Non-linear transformations (crucial for gradient boosting)
            df['prog_squared'] = df['progesterone_ng_ml'] ** 2
            df['prog_log'] = np.log1p(df['progesterone_ng_ml'])
            df['prog_cube'] = df['progesterone_ng_ml'] ** 3
            df['prog_inv'] = 1.0 / (df['progesterone_ng_ml'] + 0.1)
            
            df['lh_squared'] = df['lh_level_miu_l'] ** 2
            df['lh_log'] = np.log1p(df['lh_level_miu_l'])
            df['lh_cube'] = df['lh_level_miu_l'] ** 3
            
            # Rate of change transformations
            df['prog_roc_squared'] = df['prog_roc'] ** 2
            df['prog_roc_abs'] = np.abs(df['prog_roc'])
            df['prog_roc_cubed'] = df['prog_roc'] ** 3
            df['lh_roc_abs'] = np.abs(df['lh_roc'])
            
            # Cycle day
            df['cycle_day_squared'] = df['cycle_day'] ** 2
            df['cycle_day_sqrt'] = np.sqrt(df['cycle_day'])
            df['cycle_day_log'] = np.log1p(df['cycle_day'])
            
            # Estimates
            df['est_days_post_ov'] = np.where(
                df['progesterone_ng_ml'] >= 5.0,
                np.clip(df['cycle_day'] - 14, 0, 20),
                0
            )
            
            df['generic_cycle_estimate'] = np.clip(28 - df['cycle_day'], 0, 35)
            df['typical_luteal_length'] = 13
            
            # ============================================================
            # FEATURE SELECTION
            # ============================================================
            feature_columns = [
                # Rate of change (primary for non-linear model)
                'prog_roc',
                'prog_roc_abs',
                'prog_roc_squared',
                'prog_roc_cubed',
                'lh_roc',
                'lh_roc_abs',
                'prog_declining',
                'prog_rapid_decline',
                'lh_surging',
                'lh_post_surge_decline',
                
                # Hormone levels (raw and transformed)
                'progesterone_ng_ml',
                'lh_level_miu_l',
                'prog_squared',
                'prog_cube',
                'prog_log',
                'prog_inv',
                'lh_squared',
                'lh_cube',
                'lh_log',
                
                # Hormone states
                'prog_premenstrual_low',
                'prog_late_luteal',
                'prog_mid_luteal',
                'prog_early_luteal',
                'prog_follicular',
                'prog_ovulation_confirmed',
                'lh_surge',
                'lh_elevated',
                'lh_baseline',
                
                # Hormone interactions
                'lh_to_prog_ratio',
                'prog_to_lh_ratio',
                'hormone_product',
                'prog_lh_roc_interaction',
                
                # Cycle position
                'cycle_day',
                'cycle_day_squared',
                'cycle_day_sqrt',
                'cycle_day_log',
                'late_cycle',
                'mid_cycle',
                'early_cycle',
                
                # Combinations
                'period_warning_critical',
                'period_warning_strong',
                'post_ovulation_confirmed',
                
                # Estimates
                'est_days_post_ov',
                'generic_cycle_estimate',
                'typical_luteal_length',
                'prog_stable_or_rising',
                
                # Metadata
                'age_numeric',
                'phase_encoded',
            ]
            
            target = 'days_to_next_cycle'
            
            required_cols = feature_columns + [target] + ['age_group', 'subject_id']
            df_clean = df[required_cols].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target]
            age_groups = df_clean['age_group']
            
            valid_mask = (
                (y >= 0) & (y <= 35) & 
                (df_clean['lh_level_miu_l'] > 0) & 
                (df_clean['progesterone_ng_ml'] > 0)
            )
            X = X[valid_mask]
            y = y[valid_mask]
            age_groups = age_groups[valid_mask]
            
            self.feature_names = feature_columns
            
            print(f"\n✓ Prepared: {len(X)} samples with {len(feature_columns)} features")
            print(f"\nUsing Gradient Boosting Regressor (Non-Linear Model)")
            print(f"This captures non-linear hormone patterns and interactions")
            
            return X, y, age_groups
            
        except FileNotFoundError:
            print("CSV file not found.")
            return None, None, None
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def plot_progesterone_rate_of_change(self):
        """Visualize progesterone rate of change vs days to period"""
        if self.prog_roc_data is None:
            print("No rate of change data available. Train model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Progesterone Rate of Change Analysis', fontsize=16, fontweight='bold')
        
        df = self.prog_roc_data.dropna()
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['days_to_next_cycle'], df['progesterone_ng_ml'], 
                            c=df['cycle_day'], cmap='viridis', alpha=0.6, s=20)
        ax1.axhline(y=5.0, color='r', linestyle='--', label='Ovulation threshold (5 ng/mL)')
        ax1.axhline(y=1.5, color='orange', linestyle='--', label='Premenstrual (<1.5 ng/mL)')
        ax1.set_xlabel('Days Until Next Period', fontsize=12)
        ax1.set_ylabel('Progesterone Level (ng/mL)', fontsize=12)
        ax1.set_title('Progesterone Level vs Period Timing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cycle Day')
        
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['days_to_next_cycle'], df['prog_roc'], 
                              c=df['progesterone_ng_ml'], cmap='coolwarm', alpha=0.6, s=20)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.axhline(y=-1.0, color='orange', linestyle='--', label='Declining (-1 ng/mL)')
        ax2.axhline(y=-2.0, color='r', linestyle='--', label='Rapid decline (-2 ng/mL)')
        ax2.set_xlabel('Days Until Next Period', fontsize=12)
        ax2.set_ylabel('Progesterone Rate of Change (ng/mL/day)', fontsize=12)
        ax2.set_title('Rate of Change vs Period Timing')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Prog Level (ng/mL)')
        
        ax3 = axes[1, 0]
        bins = [0, 3, 7, 14, 21, 35]
        labels = ['0-3d', '4-7d', '8-14d', '15-21d', '22+d']
        df['days_bin'] = pd.cut(df['days_to_next_cycle'], bins=bins, labels=labels)
        df.boxplot(column='prog_roc', by='days_bin', ax=ax3)
        ax3.set_xlabel('Days Until Period (Binned)', fontsize=12)
        ax3.set_ylabel('Progesterone Rate of Change', fontsize=12)
        ax3.set_title('Rate of Change Distribution by Period Proximity')
        ax3.axhline(y=-1.0, color='orange', linestyle='--', alpha=0.7)
        ax3.axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
        plt.sca(ax3)
        plt.xticks(rotation=45)
        
        ax4 = axes[1, 1]
        df_sample = df.sample(min(500, len(df)))
        scatter4 = ax4.scatter(df_sample['progesterone_ng_ml'], df_sample['prog_roc'],
                              c=df_sample['days_to_next_cycle'], cmap='RdYlGn_r', 
                              alpha=0.7, s=30, edgecolors='k', linewidth=0.5)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label='Ovulation (5 ng/mL)')
        ax4.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='Premenstrual (1.5 ng/mL)')
        ax4.set_xlabel('Progesterone Level (ng/mL)', fontsize=12)
        ax4.set_ylabel('Rate of Change (ng/mL/day)', fontsize=12)
        ax4.set_title('Progesterone Trajectory Space')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter4, ax=ax4, label='Days to Period')
        
        plt.tight_layout()
        plt.savefig('progesterone_rate_of_change_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'progesterone_rate_of_change_analysis.png'")
        plt.show()
    
    def stratified_train_test_split(self, X, y, age_groups, test_size=0.2, random_state=42):
        """Stratified sampling"""
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        
        print(f"\nStratified split (80% train, 20% test):")
        print("-" * 60)
        
        for age_group in age_groups.unique():
            age_mask = age_groups == age_group
            X_age, y_age = X[age_mask], y[age_mask]
            
            if len(X_age) < 2:
                continue
                
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_age, y_age, test_size=test_size, random_state=random_state, shuffle=True
            )
            
            X_train_list.append(X_tr)
            X_test_list.append(X_te)
            y_train_list.append(y_tr)
            y_test_list.append(y_te)
            
            print(f"{age_group}: {len(X_age)} → {len(X_tr)} train, {len(X_te)} test")
        
        X_train = pd.concat(X_train_list, ignore_index=True)
        X_test = pd.concat(X_test_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        y_test = pd.concat(y_test_list, ignore_index=True)
        
        print(f"\nTotal: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    def calculate_accuracy_metrics(self, y_true, y_pred):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        errors = np.abs(y_true - y_pred)
        within_1 = np.mean(errors <= 1.0) * 100
        within_2 = np.mean(errors <= 2.0) * 100
        within_3 = np.mean(errors <= 3.0) * 100
        within_5 = np.mean(errors <= 5.0) * 100
        exact = (np.sum(np.round(y_pred) == np.round(y_true)) / len(y_true)) * 100
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"MAE:  {mae:.2f} days  |  RMSE: {rmse:.2f} days  |  R²: {r2:.3f}")
        print(f"\nAccuracy:")
        print(f"  Exact:    {exact:.1f}%  |  ±1 day: {within_1:.1f}%")
        print(f"  ±2 days:  {within_2:.1f}%  |  ±3 days: {within_3:.1f}%")
        print(f"  ±5 days:  {within_5:.1f}%")
        print(f"{'='*60}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'exact': exact,
                'within_1': within_1, 'within_2': within_2}
    
    def train_model(self, X, y, age_groups):
        """Train gradient boosting model"""
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(X, y, age_groups)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING MODEL")
        print("(Captures non-linear hormone patterns)")
        print("="*60)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = self.calculate_accuracy_metrics(y_test, y_pred)
        self.is_trained = True
        
        print(f"\nTop 20 Features (by importance):")
        print("-" * 60)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(20).iterrows():
            bar = '█' * int(min(row['importance'] * 500, 40))
            print(f"{row['feature']:35} {bar} {row['importance']:.4f}")
        
        return metrics
    
    def predict_next_period(self, age, lh_level, progesterone_level, current_cycle_day=None,
                           prev_prog_level=None):
        """Make prediction"""
        if not self.is_trained:
            print("Model not trained.")
            return None
        
        if current_cycle_day is None:
            if progesterone_level > 5.0:
                current_cycle_day = 20
            elif lh_level > 10.0:
                current_cycle_day = 14
            else:
                current_cycle_day = 10
        
        if prev_prog_level is not None:
            prog_roc = progesterone_level - prev_prog_level
        else:
            prog_roc = 0
        
        lh_roc = 0
        
        if progesterone_level > 4.0:
            phase = 'luteal'
        elif lh_level > 10.0:
            phase = 'ovulatory'
        else:
            phase = 'follicular'
        
        phase_encoded = self.label_encoder.transform([phase])[0]
        
        # Calculate all features
        prog_roc_abs = abs(prog_roc)
        prog_roc_squared = prog_roc ** 2
        prog_roc_cubed = prog_roc ** 3
        lh_roc_abs = abs(lh_roc)
        prog_declining = int(prog_roc < -1.0)
        prog_rapid_decline = int(prog_roc < -2.0)
        lh_surging = 0
        lh_post_surge_decline = 0
        
        progesterone_sq = progesterone_level ** 2
        progesterone_cube = progesterone_level ** 3
        progesterone_log = np.log1p(progesterone_level)
        progesterone_inv = 1.0 / (progesterone_level + 0.1)
        
        lh_squared = lh_level ** 2
        lh_cube = lh_level ** 3
        lh_log = np.log1p(lh_level)
        
        prog_premenstrual_low = int(progesterone_level < 1.5)
        prog_late_luteal = int(1.5 <= progesterone_level < 5.0)
        prog_mid_luteal = int(10.0 <= progesterone_level <= 25.0)
        prog_early_luteal = int(5.0 <= progesterone_level < 10.0)
        prog_follicular = int(progesterone_level < 2.0)
        prog_ovulation_confirmed = int(progesterone_level >= 5.0)
        
        lh_surge = int(lh_level > 10.0)
        lh_elevated = int(7.0 < lh_level <= 10.0)
        lh_baseline = int(lh_level <= 7.0)
        
        lh_to_prog_ratio = lh_level / (progesterone_level + 0.1)
        prog_to_lh_ratio = progesterone_level / (lh_level + 0.1)
        hormone_product = lh_level * progesterone_level
        prog_lh_roc_interaction = prog_roc * lh_roc
        
        cycle_day_squared = current_cycle_day ** 2
        cycle_day_sqrt = np.sqrt(current_cycle_day)
        cycle_day_log = np.log1p(current_cycle_day)
        
        late_cycle = int(current_cycle_day > 20)
        mid_cycle = int(10 < current_cycle_day <= 20)
        early_cycle = int(current_cycle_day <= 10)
        
        period_warning_critical = int(late_cycle and prog_rapid_decline and progesterone_level < 3.0)
        period_warning_strong = int(late_cycle and prog_declining and progesterone_level < 5.0)
        post_ovulation_confirmed = int(mid_cycle and 10.0 <= progesterone_level <= 25.0)
        
        est_days_post_ov = max(0, current_cycle_day - 14) if progesterone_level >= 5.0 else 0
        generic_cycle_estimate = max(0, 28 - current_cycle_day)
        typical_luteal_length = 13
        prog_stable_or_rising = int(prog_roc >= 0)
        
        features = np.array([[
            prog_roc, prog_roc_abs, prog_roc_squared, prog_roc_cubed, lh_roc, lh_roc_abs,
            prog_declining, prog_rapid_decline, lh_surging, lh_post_surge_decline,
            progesterone_level, lh_level, progesterone_sq, progesterone_cube, progesterone_log, progesterone_inv,
            lh_squared, lh_cube, lh_log,
            prog_premenstrual_low, prog_late_luteal, prog_mid_luteal, prog_early_luteal,
            prog_follicular, prog_ovulation_confirmed, lh_surge, lh_elevated, lh_baseline,
            lh_to_prog_ratio, prog_to_lh_ratio, hormone_product, prog_lh_roc_interaction,
            current_cycle_day, cycle_day_squared, cycle_day_sqrt, cycle_day_log,
            late_cycle, mid_cycle, early_cycle,
            period_warning_critical, period_warning_strong, post_ovulation_confirmed,
            est_days_post_ov, generic_cycle_estimate, typical_luteal_length, prog_stable_or_rising,
            age, phase_encoded
        ]])
        
        features_scaled = self.scaler.transform(features)
        days_until = float(self.model.predict(features_scaled)[0])
        days_until = max(0, min(35, days_until))
        
        predicted_date = datetime.now() + timedelta(days=int(days_until))
        
        if period_warning_critical:
            confidence = 95
        elif period_warning_strong:
            confidence = 90
        elif post_ovulation_confirmed:
            confidence = 85
        elif prog_ovulation_confirmed:
            confidence = 80
        else:
            confidence = max(65, 100 - abs(28 - days_until) * 2)
        
        return {
            'days_until_next_period': round(days_until, 1),
            'predicted_date': predicted_date.strftime('%Y-%m-%d'),
            'current_phase': phase,
            'confidence_score': round(confidence, 1),
            'prog_rate_of_change': round(prog_roc, 2),
            'input_summary': {
                'age': age,
                'lh': lh_level,
                'progesterone': progesterone_level,
                'cycle_day': current_cycle_day,
                'prog_roc': round(prog_roc, 2)
            }
        }
    
    def save_model(self, filename='ultimate_cycle_model.pkl'):
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, filename)
            print(f"\n✓ Model saved as '{filename}'")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ULTIMATE MENSTRUAL CYCLE PREDICTION MODEL")
    print("Non-Linear Gradient Boosting (Captures Hormone Curves)")
    print("="*70)
    
    model = UltimateMenstrualCycleModel()
    X, y, age_groups = model.load_and_prepare_data('DataSet.csv')
    
    if X is not None:
        metrics = model.train_model(X, y, age_groups)
        model.save_model()
        
        print("\nGenerating rate of change visualization...")
        model.plot_progesterone_rate_of_change()
        
        print("\n" + "="*70)
        print("INTERACTIVE PREDICTION")
        print("="*70)
        
        while True:
            try:
                print("\nEnter data (or 'q' to quit):")
                age_input = input("Age: ")
                if age_input.lower() == "q":
                    break
                age = int(age_input)
                
                lh = float(input("LH (mIU/L): "))
                prog = float(input("Progesterone (ng/mL): "))
                
                cycle_day_input = input("Cycle day (blank = estimate): ")
                cycle_day = int(cycle_day_input) if cycle_day_input.strip() else None
                
                prev_prog_input = input("Previous progesterone (blank = skip): ")
                prev_prog = float(prev_prog_input) if prev_prog_input.strip() else None
                
                result = model.predict_next_period(age, lh, prog, cycle_day, prev_prog)
                
                if result:
                    print("\n" + "="*50)
                    print("PREDICTION")
                    print("="*50)
                    print(f"Next period:  {result['days_until_next_period']} days")
                    print(f"Date:         {result['predicted_date']}")
                    print(f"Confidence:   {result['confidence_score']}%")
                    print(f"Phase:        {result['current_phase']}")
                    if prev_prog:
                        print(f"Prog change:  {result['prog_rate_of_change']} ng/mL/day")
                    print("="*50)
                    
            except ValueError:
                print("Invalid input.")
            except Exception as e:
                print(f"Error: {e}")