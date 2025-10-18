import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib

class UltimateMenstrualCycleModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.prog_roc_data = None  # Store rate of change data for visualization
        
    def calculate_rate_of_change(self, df):
        """
        Calculate rate of change for progesterone and LH
        This is KEY - declining progesterone predicts period timing!
        """
        # Sort by subject and cycle day to ensure proper order
        df = df.sort_values(['subject_id', 'cycle_day'])
        
        # Calculate rate of change (difference from previous measurement)
        df['prog_roc'] = df.groupby('subject_id')['progesterone_ng_ml'].diff()
        df['lh_roc'] = df.groupby('subject_id')['lh_level_miu_l'].diff()
        
        # Fill NaN with 0 (first measurement has no previous value)
        df['prog_roc'] = df['prog_roc'].fillna(0)
        df['lh_roc'] = df['lh_roc'].fillna(0)
        
        # CRITICAL FEATURES from research:
        # Progesterone drops "a couple days before menstruation"
        # Rapid decline = period imminent
        df['prog_declining'] = (df['prog_roc'] < -1.0).astype(int)  # Dropping >1 ng/mL
        df['prog_rapid_decline'] = (df['prog_roc'] < -2.0).astype(int)  # Dropping >2 ng/mL
        df['prog_stable_or_rising'] = (df['prog_roc'] >= 0).astype(int)
        
        # LH surge and decline patterns
        df['lh_surging'] = (df['lh_roc'] > 2.0).astype(int)  # Rapid LH increase
        df['lh_post_surge_decline'] = (df['lh_roc'] < -2.0).astype(int)
        
        return df
    
    def load_and_prepare_data(self, csv_file='DataSet.csv'):
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded data: {len(df)} samples")
            
            # Age mapping
            age_mapping = {'25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42}
            df['age_numeric'] = df['age_group'].map(age_mapping)
            
            # Phase encoding
            df['phase_encoded'] = self.label_encoder.fit_transform(df['phase'])
            
            # ============================================================
            # RATE OF CHANGE FEATURES (MOST IMPORTANT!)
            # ============================================================
            print("\nCalculating rate of change features...")
            df = self.calculate_rate_of_change(df)
            
            # Store for visualization
            self.prog_roc_data = df[['cycle_day', 'progesterone_ng_ml', 'prog_roc', 
                                     'days_to_next_cycle']].copy()
            
            # ============================================================
            # RESEARCH-BASED HORMONE THRESHOLDS
            # From search: Peaks 10-25 ng/mL at 6-8 days post-ovulation
            # Drops to <1 ng/mL before period
            # ============================================================
            
            # Progesterone absolute levels (research-backed)
            df['prog_premenstrual_low'] = (df['progesterone_ng_ml'] < 1.5).astype(int)  # <1.5 = period very soon
            df['prog_late_luteal'] = ((df['progesterone_ng_ml'] >= 1.5) & 
                                      (df['progesterone_ng_ml'] < 5.0)).astype(int)  # Declining phase
            df['prog_mid_luteal'] = ((df['progesterone_ng_ml'] >= 10.0) & 
                                     (df['progesterone_ng_ml'] <= 25.0)).astype(int)  # Peak range
            df['prog_early_luteal'] = ((df['progesterone_ng_ml'] >= 5.0) & 
                                       (df['progesterone_ng_ml'] < 10.0)).astype(int)
            df['prog_follicular'] = (df['progesterone_ng_ml'] < 2.0).astype(int)
            df['prog_ovulation_confirmed'] = (df['progesterone_ng_ml'] >= 5.0).astype(int)  # Clinical threshold
            
            # LH patterns
            df['lh_surge'] = (df['lh_level_miu_l'] > 10.0).astype(int)  # Ovulation happening
            df['lh_elevated'] = ((df['lh_level_miu_l'] > 7.0) & 
                                 (df['lh_level_miu_l'] <= 10.0)).astype(int)
            df['lh_baseline'] = (df['lh_level_miu_l'] <= 7.0).astype(int)
            
            # ============================================================
            # CRITICAL COMBINATIONS (Late cycle + hormone patterns)
            # ============================================================
            df['late_cycle'] = (df['cycle_day'] > 20).astype(int)
            df['mid_cycle'] = ((df['cycle_day'] > 10) & (df['cycle_day'] <= 20)).astype(int)
            df['early_cycle'] = (df['cycle_day'] <= 10).astype(int)
            
            # HIGHEST PRIORITY: Late cycle + declining progesterone = PERIOD IMMINENT
            df['period_warning_strong'] = ((df['late_cycle'] == 1) & 
                                          (df['prog_declining'] == 1) & 
                                          (df['progesterone_ng_ml'] < 5.0)).astype(int)
            
            df['period_warning_critical'] = ((df['late_cycle'] == 1) & 
                                            (df['prog_rapid_decline'] == 1) & 
                                            (df['progesterone_ng_ml'] < 3.0)).astype(int)
            
            # Mid-cycle + high prog = Post-ovulation (13 days to period from research)
            df['post_ovulation_confirmed'] = ((df['mid_cycle'] == 1) & 
                                              (df['prog_mid_luteal'] == 1)).astype(int)
            
            # ============================================================
            # HORMONE DYNAMICS (Ratios and Interactions)
            # ============================================================
            df['lh_to_prog_ratio'] = df['lh_level_miu_l'] / (df['progesterone_ng_ml'] + 0.1)
            df['prog_to_lh_ratio'] = df['progesterone_ng_ml'] / (df['lh_level_miu_l'] + 0.1)
            df['hormone_product'] = df['lh_level_miu_l'] * df['progesterone_ng_ml']
            
            # Rate of change ratios
            df['prog_lh_roc_interaction'] = df['prog_roc'] * df['lh_roc']
            
            # ============================================================
            # NON-LINEAR TRANSFORMATIONS
            # ============================================================
            df['prog_squared'] = df['progesterone_ng_ml'] ** 2
            df['prog_log'] = np.log1p(df['progesterone_ng_ml'])
            df['lh_squared'] = df['lh_level_miu_l'] ** 2
            df['cycle_day_squared'] = df['cycle_day'] ** 2
            df['cycle_day_sqrt'] = np.sqrt(df['cycle_day'])
            
            # Rate of change transformations
            df['prog_roc_squared'] = df['prog_roc'] ** 2
            df['prog_roc_abs'] = np.abs(df['prog_roc'])
            
            # ============================================================
            # WEAK INDICATORS (Low priority - let data dominate)
            # ============================================================
            # Research: Luteal phase = 13 days (stable), but use as MINOR feature
            df['typical_luteal_length'] = 13
            
            # Generic cycle assumption (VERY LOW WEIGHT)
            df['generic_cycle_estimate'] = np.clip(28 - df['cycle_day'], 0, 35)
            
            # If ovulated (prog ≥5), estimate days post-ovulation
            df['est_days_post_ov'] = np.where(
                df['progesterone_ng_ml'] >= 5.0,
                np.clip(df['cycle_day'] - 14, 0, 20),
                0
            )
            
            # ============================================================
            # FEATURE SELECTION - PRIORITIZED BY IMPORTANCE
            # ============================================================
            feature_columns = [
                # TIER 1: RATE OF CHANGE (Most predictive from data)
                'prog_roc',                    # Raw rate of change
                'prog_roc_abs',                # Magnitude of change
                'prog_roc_squared',            # Non-linear rate
                'prog_declining',              # Boolean: declining
                'prog_rapid_decline',          # Boolean: rapid decline
                'lh_roc',
                
                # TIER 2: CRITICAL COMBINATIONS (Data + Research)
                'period_warning_critical',     # Late + rapid prog decline
                'period_warning_strong',       # Late + prog declining
                'post_ovulation_confirmed',    # Mid-cycle + high prog
                
                # TIER 3: ABSOLUTE HORMONE LEVELS (Direct measurements)
                'progesterone_ng_ml',
                'lh_level_miu_l',
                'prog_premenstrual_low',
                'prog_late_luteal',
                'prog_mid_luteal',
                'prog_early_luteal',
                'prog_ovulation_confirmed',
                
                # TIER 4: CYCLE POSITION
                'cycle_day',
                'late_cycle',
                'mid_cycle',
                'early_cycle',
                
                # TIER 5: LH PATTERNS
                'lh_surge',
                'lh_elevated',
                'lh_baseline',
                'lh_surging',
                'lh_post_surge_decline',
                
                # TIER 6: HORMONE DYNAMICS
                'lh_to_prog_ratio',
                'prog_to_lh_ratio',
                'hormone_product',
                'prog_lh_roc_interaction',
                
                # TIER 7: NON-LINEAR TRANSFORMS
                'prog_squared',
                'prog_log',
                'lh_squared',
                'cycle_day_squared',
                'cycle_day_sqrt',
                'prog_stable_or_rising',
                'prog_follicular',
                
                # TIER 8: RESEARCH-BASED ESTIMATES (Minor weight)
                'est_days_post_ov',
                'typical_luteal_length',
                
                # TIER 9: GENERIC (Weakest - let model decide)
                'generic_cycle_estimate',
                
                # METADATA
                'age_numeric',
                'phase_encoded',
            ]
            
            target = 'days_to_next_cycle'
            
            # Clean data
            required_cols = feature_columns + [target] + ['age_group', 'subject_id']
            df_clean = df[required_cols].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target]
            age_groups = df_clean['age_group']
            
            # Validation
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
            print(f"\nFeature tiers:")
            print(f"  Tier 1: Rate of change (6 features)")
            print(f"  Tier 2: Critical combinations (3 features)")
            print(f"  Tier 3-9: Hormone levels, patterns, estimates")
            print(f"  Total: {len(feature_columns)} features")
            
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
        
        # Plot 1: Progesterone level vs Days to Period
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
        
        # Plot 2: Rate of Change vs Days to Period
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
        
        # Plot 3: Distribution of Rate of Change by Days to Period
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
        
        # Plot 4: Heatmap of Prog Level vs ROC colored by Days to Period
        ax4 = axes[1, 1]
        # Create bins for visualization
        df_sample = df.sample(min(500, len(df)))  # Sample for clearer visualization
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
        
        # Add quadrant labels
        ax4.text(15, 2, 'Rising\n(Early Luteal)', ha='center', va='center', 
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax4.text(15, -2, 'Declining\n(Late Luteal)', ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax4.text(1, -2, 'Low & Falling\n(Imminent)', ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
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
        """Train model"""
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(X, y, age_groups)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n" + "="*60)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("="*60)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = self.calculate_accuracy_metrics(y_test, y_pred)
        self.is_trained = True
        
        # Feature importance
        print(f"\nTop 20 Features (by coefficient magnitude):")
        print("-" * 60)
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        for idx, row in coef_df.head(20).iterrows():
            direction = "↓" if row['coefficient'] < 0 else "↑"
            bar = '█' * int(min(abs(row['coefficient']) * 3, 40))
            print(f"{row['feature']:35} {bar} {row['coefficient']:7.3f} {direction}")
        
        return metrics
    
    def predict_next_period(self, age, lh_level, progesterone_level, current_cycle_day=None,
                           prev_prog_level=None):
        """
        Make prediction with optional previous progesterone for rate of change
        """
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
        
        # Calculate rate of change if previous measurement provided
        if prev_prog_level is not None:
            prog_roc = progesterone_level - prev_prog_level
        else:
            prog_roc = 0  # Default if no previous measurement
        
        lh_roc = 0  # Would need previous LH to calculat
        
        # Determine phase
        if progesterone_level > 4.0:
            phase = 'luteal'
        elif lh_level > 10.0:
            phase = 'ovulatory'
        else:
            phase = 'follicular'
        
        phase_encoded = self.label_encoder.transform([phase])[0]
        
        # Calculate all features (matching training order)
        prog_roc_abs = abs(prog_roc)
        prog_roc_squared = prog_roc ** 2
        prog_declining = int(prog_roc < -1.0)
        prog_rapid_decline = int(prog_roc < -2.0)
        
        late_cycle = int(current_cycle_day > 20)
        mid_cycle = int(10 < current_cycle_day <= 20)
        early_cycle = int(current_cycle_day <= 10)
        
        period_warning_critical = int(late_cycle and prog_rapid_decline and progesterone_level < 3.0)
        period_warning_strong = int(late_cycle and prog_declining and progesterone_level < 5.0)
        post_ovulation_confirmed = int(mid_cycle and 10.0 <= progesterone_level <= 25.0)
        
        prog_premenstrual_low = int(progesterone_level < 1.5)
        prog_late_luteal = int(1.5 <= progesterone_level < 5.0)
        prog_mid_luteal = int(10.0 <= progesterone_level <= 25.0)
        prog_early_luteal = int(5.0 <= progesterone_level < 10.0)
        prog_ovulation_confirmed = int(progesterone_level >= 5.0)
        
        lh_surge = int(lh_level > 10.0)
        lh_elevated = int(7.0 < lh_level <= 10.0)
        lh_baseline = int(lh_level <= 7.0)
        lh_surging = 0
        lh_post_surge_decline = 0
        
        lh_to_prog_ratio = lh_level / (progesterone_level + 0.1)
        prog_to_lh_ratio = progesterone_level / (lh_level + 0.1)
        hormone_product = lh_level * progesterone_level
        prog_lh_roc_interaction = prog_roc * lh_roc
        
        prog_squared = progesterone_level ** 2
        prog_log = np.log1p(progesterone_level)
        lh_squared = lh_level ** 2
        cycle_day_squared = current_cycle_day ** 2
        cycle_day_sqrt = np.sqrt(current_cycle_day)
        prog_stable_or_rising = int(prog_roc >= 0)
        prog_follicular = int(progesterone_level < 2.0)
        
        est_days_post_ov = max(0, current_cycle_day - 14) if progesterone_level >= 5.0 else 0
        typical_luteal_length = 13
        generic_cycle_estimate = max(0, 28 - current_cycle_day)
        
        # Build feature array
        features = np.array([[
            prog_roc, prog_roc_abs, prog_roc_squared, prog_declining, prog_rapid_decline, lh_roc,
            period_warning_critical, period_warning_strong, post_ovulation_confirmed,
            progesterone_level, lh_level, prog_premenstrual_low, prog_late_luteal,
            prog_mid_luteal, prog_early_luteal, prog_ovulation_confirmed,
            current_cycle_day, late_cycle, mid_cycle, early_cycle,
            lh_surge, lh_elevated, lh_baseline, lh_surging, lh_post_surge_decline,
            lh_to_prog_ratio, prog_to_lh_ratio, hormone_product, prog_lh_roc_interaction,
            prog_squared, prog_log, lh_squared, cycle_day_squared, cycle_day_sqrt,
            prog_stable_or_rising, prog_follicular,
            est_days_post_ov, typical_luteal_length, generic_cycle_estimate,
            age, phase_encoded
        ]])
        
        features_scaled = self.scaler.transform(features)
        days_until = float(self.model.predict(features_scaled)[0])
        days_until = max(0, min(35, days_until))
        
        predicted_date = datetime.now() + timedelta(days=int(days_until))
        
        # Smart confidence based on rate of change
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
    print("Rate of Change + Research-Based Features")
    print("="*70)
    
    model = UltimateMenstrualCycleModel()
    X, y, age_groups = model.load_and_prepare_data('DataSet.csv')
    
    if X is not None:
        metrics = model.train_model(X, y, age_groups)
        model.save_model()
        
        # Generate visualization
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
    else:
        print("Could not load data.")