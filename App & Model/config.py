# ===============================
# CONFIGURATION FILE
# ===============================
# This file contains all configuration settings for the application
# To revert changes, simply set ENABLE_RECOMMENDATIONS = False

# ===============================
# FEATURE FLAGS
# ===============================
ENABLE_RECOMMENDATIONS = True  # Set to False to disable recommendation features
ENABLE_ORANGE_DATASET = False  # Set to True to enable Orange dataset integration

# ===============================
# DATASET PATHS
# ===============================
# IBM Telco Customer Churn Dataset
IBM_DATA_PATH = r"C:\Users\Mazhar Iqbal\Churn_Projects_Sample\Customer-Churn-Prediction-Dashboard\Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Orange Dataset Paths (optional - for enhanced training)
ORANGE_DATA_PATHS = [
    # Uncomment and modify these paths if you have Orange datasets
    # r"C:\Users\Mazhar Iqbal\Churn_Projects_Sample\Customer-Churn-Prediction-Dashboard\Dataset\churn-bigml-20.csv",
    # r"C:\Users\Mazhar Iqbal\Churn_Projects_Sample\Customer-Churn-Prediction-Dashboard\Dataset\churn-bigml-80.csv"
]

# Model Paths
MODEL_PATH = r"C:\Users\Mazhar Iqbal\Churn_Projects_Sample\Customer-Churn-Prediction-Dashboard\App & Model\churn_model.pkl"
RECOMMENDATION_MODEL_PATH = r"C:\Users\Mazhar Iqbal\Churn_Projects_Sample\Customer-Churn-Prediction-Dashboard\App & Model\recommendation_model.pkl"

# ===============================
# RECOMMENDATION ENGINE SETTINGS
# ===============================
RECOMMENDATION_TOP_K_FEATURES = 5
RECOMMENDATION_MAX_REASONS = 3
RECOMMENDATION_MAX_OFFERS = 4

# ===============================
# MODEL TRAINING SETTINGS
# ===============================
LGBM_N_ESTIMATORS = 200
LGBM_LEARNING_RATE = 0.05
LGBM_CLASS_WEIGHT = "balanced"
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# ===============================
# UI SETTINGS
# ===============================
CHURN_COLOR_MAP = {"Yes": "#4E9F3D", "No": "#FF0000"}
HEATMAP_SCALE = "Turbo"
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLOR_SEQUENCE = "qualitative.Set3"

# ===============================
# CACHE SETTINGS
# ===============================
ENABLE_CACHING = True
CACHE_TTL = 3600  # 1 hour in seconds
