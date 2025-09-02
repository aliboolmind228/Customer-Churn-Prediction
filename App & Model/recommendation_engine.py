import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import shap
import random

# ===============================
# FEATURE MAPPING FOR RECOMMENDATIONS
# ===============================

def create_feature_mapping():
    """Create mapping based on actual dataset features"""
    return {
        # IBM Dataset Features
        'Contract': {
            'reasons': ['Month-to-month risk', 'Contract flexibility needed', 'Commitment concerns'],
            'offers': ['Annual contract discount', 'Flexible payment terms', 'Contract loyalty bonus']
        },
        'tenure': {
            'reasons': ['New customer risk', 'Short relationship', 'Loyalty building needed'],
            'offers': ['New customer bonus', 'Loyalty rewards', 'Retention discount']
        },
        'MonthlyCharges': {
            'reasons': ['High monthly cost', 'Price sensitivity', 'Budget constraints'],
            'offers': ['Monthly discount', 'Price reduction', 'Budget plan option']
        },
        'TotalCharges': {
            'reasons': ['High total spending', 'Cost accumulation concern', 'Value perception'],
            'offers': ['Total bill discount', 'Value package upgrade', 'Spending rewards']
        },
        'OnlineSecurity': {
            'reasons': ['No security service', 'Security concerns', 'Missing protection'],
            'offers': ['Free security upgrade', 'Premium protection plan', 'Identity monitoring']
        },
        'TechSupport': {
            'reasons': ['No tech support', 'Support needs unmet', 'Technical difficulties'],
            'offers': ['Free tech support', 'Priority assistance', '24/7 help desk']
        },
        'InternetService': {
            'reasons': ['Basic internet plan', 'Speed limitations', 'Service quality issues'],
            'offers': ['Speed upgrade', 'Fiber promotion', 'Enhanced connectivity']
        },
        'StreamingTV': {
            'reasons': ['No TV streaming', 'Entertainment needs unmet', 'Content limitations'],
            'offers': ['Free TV package', 'Premium channels', 'Entertainment bundle']
        },
        'StreamingMovies': {
            'reasons': ['No movie streaming', 'Limited content access', 'Entertainment gaps'],
            'offers': ['Movie package deal', 'Premium content access', 'Streaming discount']
        },
        'PaymentMethod': {
            'reasons': ['Manual payment hassle', 'Payment inconvenience', 'Billing complexity'],
            'offers': ['Auto-pay discount', 'Payment convenience bonus', 'Cashback rewards']
        },
        'PaperlessBilling': {
            'reasons': ['Paper billing preference', 'Digital adoption gap', 'Billing method issues'],
            'offers': ['Paperless incentive', 'Digital discount', 'Environmental reward']
        },
        
        # Orange Dataset Features  
        'Customer service calls': {
            'reasons': ['Frequent support needs', 'Service quality issues', 'Unresolved problems'],
            'offers': ['Priority support', 'Dedicated agent', 'Service improvement plan']
        },
        'International plan': {
            'reasons': ['No international service', 'Global communication needs', 'Travel limitations'],
            'offers': ['International package', 'Global calling plan', 'Travel benefits']
        },
        'Account length': {
            'reasons': ['Short account history', 'New relationship', 'Trust building needed'],
            'offers': ['Account building bonus', 'History rewards', 'Stability incentive']
        },
        'Total day minutes': {
            'reasons': ['High daytime usage', 'Peak hour costs', 'Usage pattern mismatch'],
            'offers': ['Unlimited day calls', 'Peak hour discount', 'Usage optimization']
        },
        'Total eve minutes': {
            'reasons': ['High evening usage', 'After-hours needs', 'Time-based costs'],
            'offers': ['Evening plan upgrade', 'Off-peak pricing', 'Night call bonus']
        },
        'Total night minutes': {
            'reasons': ['High nighttime usage', 'Late-hour communication', 'Night rate costs'],
            'offers': ['Night plan optimization', '24-hour unlimited', 'Late night discount']
        },
        'Total intl minutes': {
            'reasons': ['High international usage', 'Global communication costs', 'Cross-border needs'],
            'offers': ['International bundle', 'Global minute package', 'Country-specific deals']
        },
        'Voice mail plan': {
            'reasons': ['Basic voicemail limits', 'Message management needs', 'Communication gaps'],
            'offers': ['Voicemail upgrade', 'Advanced features', 'Message management tools']
        },
        'Number vmail messages': {
            'reasons': ['Heavy voicemail usage', 'Message overflow', 'Storage limitations'],
            'offers': ['Enhanced voicemail', 'Storage upgrade', 'Message transcription']
        }
    }

# ===============================
# RECOMMENDATION ENGINE CLASS
# ===============================

class SimpleRecommendationEngine:
    
    def __init__(self):
        self.feature_mapping = create_feature_mapping()
        self.model = None
        self.explainer = None
        self.feature_names = None
    
    def train_model(self, data):
        """Train the churn prediction model"""
        print("Training churn prediction model...")
        
        # Prepare data
        y = data["Churn"]
        X = data.drop(columns=["Churn", "customerID"], errors="ignore")
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
        
        lgbm.fit(X_train, y_train)
        self.model = CalibratedClassifierCV(lgbm, cv=3, method="isotonic")
        self.model.fit(X_train, y_train)
        
        # Setup SHAP
        fitted_lgbm = self.model.estimator
        self.explainer = shap.TreeExplainer(fitted_lgbm)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Model AUC: {auc:.4f}")
        
        return X_test, y_test
    
    def prepare_data_for_prediction(self, X_sample):
        """Prepare data to match training features exactly"""
        if self.feature_names is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Check data types and convert if needed
        X_sample_clean = X_sample.copy()
        for col in X_sample_clean.columns:
            if X_sample_clean[col].dtype == 'object':
                try:
                    X_sample_clean[col] = pd.to_numeric(X_sample_clean[col], errors='coerce')
                    X_sample_clean[col] = X_sample_clean[col].fillna(0)
                except:
                    print(f"Warning: Could not convert column {col} to numeric, filling with 0")
                    X_sample_clean[col] = 0
        
        # Create a DataFrame with the exact same columns as training data
        X_prepared = pd.DataFrame(0, index=X_sample_clean.index, columns=self.feature_names)
        
        # Fill in values for matching columns
        for col in self.feature_names:
            if col in X_sample_clean.columns:
                X_prepared[col] = X_sample_clean[col]
            # If column doesn't exist, it remains 0 (default value)
        
        # Debug information
        print(f"Training features: {len(self.feature_names)}")
        print(f"Sample features: {len(X_sample_clean.columns)}")
        print(f"Prepared features: {len(X_prepared.columns)}")
        print(f"Sample data types: {X_sample_clean.dtypes.value_counts().to_dict()}")
        
        return X_prepared
    
    def get_recommendations(self, X_sample, top_k=5):
        """Generate recommendations for customers"""
        
        try:
            # Prepare data to match training features exactly
            X_prepared = self.prepare_data_for_prediction(X_sample)
            
            # Get predictions
            predictions = self.model.predict(X_prepared)
            probabilities = self.model.predict_proba(X_prepared)[:, 1]
            
            # Get SHAP explanations
            shap_values = self.explainer.shap_values(X_prepared)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            recommendations = []
            
            for i in range(len(X_prepared)):
                # Get top features for this customer
                customer_shap = shap_values[i]
                top_indices = np.argsort(-np.abs(customer_shap))[:top_k]
                top_features = [self.feature_names[idx] for idx in top_indices]
                top_shap_vals = [customer_shap[idx] for idx in top_indices]
                
                # Generate reasons and offers
                reasons = []
                offers = []
                
                for feat, shap_val in zip(top_features, top_shap_vals):
                    if feat in self.feature_mapping:
                        # Get random reason and offers for this feature
                        feat_reasons = self.feature_mapping[feat]['reasons']
                        feat_offers = self.feature_mapping[feat]['offers']
                        
                        reasons.append(random.choice(feat_reasons))
                        offers.extend(random.sample(feat_offers, min(2, len(feat_offers))))
                
                # Remove duplicates and limit
                reasons = list(dict.fromkeys(reasons))[:3]
                offers = list(dict.fromkeys(offers))[:4]
                
                # If no mapping found, use generic
                if not reasons:
                    reasons = ['Service optimization needed']
                    offers = ['Personalized plan review']
                
                recommendations.append({
                    'prediction': 'Yes' if predictions[i] == 1 else 'No',
                    'prediction_proba': probabilities[i],
                    'top_features': top_features,
                    'reasons': reasons,
                    'retention_offers': offers
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            print(f"X_sample shape: {X_sample.shape}")
            print(f"X_sample columns: {X_sample.columns.tolist()}")
            if hasattr(self, 'feature_names'):
                print(f"Training feature names: {self.feature_names}")
            raise e

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_combined_datasets(ibm_path, orange_paths=None):
    """Load and combine datasets for training"""
    try:
        # Load IBM dataset
        ibm = pd.read_csv(ibm_path)
        ibm["Churn"] = ibm["Churn"].map({"Yes": 1, "No": 0})
        
        # Fix TotalCharges
        ibm["TotalCharges"] = pd.to_numeric(ibm["TotalCharges"], errors="coerce")
        ibm["TotalCharges"] = ibm["TotalCharges"].fillna(ibm["TotalCharges"].median())
        
        # Load Orange datasets if provided
        if orange_paths:
            orange_datasets = []
            for path in orange_paths:
                try:
                    orange_data = pd.read_csv(path)
                    orange_data["Churn"] = orange_data["Churn"].map({True: 1, False: 0})
                    orange_datasets.append(orange_data)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
            
            if orange_datasets:
                orange = pd.concat(orange_datasets, ignore_index=True)
                
                # Encode categorical variables for Orange dataset
                for col in orange.select_dtypes(include=["object", "bool"]).columns:
                    le = LabelEncoder()
                    orange[col] = orange[col].astype(str)
                    orange[col] = le.fit_transform(orange[col])
                
                # Combine datasets
                combined = pd.concat([ibm, orange], axis=0, ignore_index=True, sort=False)
                print(f"Combined dataset shape: {combined.shape}")
                return combined
        
        # If no Orange datasets, return IBM only
        print(f"IBM dataset shape: {ibm.shape}")
        return ibm
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None

def encode_categorical_variables(data):
    """Encode categorical variables for model training"""
    try:
        encoded_data = data.copy()
        
        for col in encoded_data.select_dtypes(include=["object", "bool"]).columns:
            if col != "customerID":
                le = LabelEncoder()
                encoded_data[col] = encoded_data[col].astype(str)
                encoded_data[col] = le.fit_transform(encoded_data[col])
        
        return encoded_data
    except Exception as e:
        print(f"Error encoding variables: {e}")
        return data
