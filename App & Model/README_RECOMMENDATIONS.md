# Customer Churn Prediction Dashboard - Recommendations System

## 🎯 Overview

This enhancement adds a powerful recommendation engine to your existing Customer Churn Prediction Dashboard. The system provides personalized retention strategies for each customer based on their risk profile and contributing factors.

## ✨ New Features

### 1. **Recommendations Tab**
- **Personalized Risk Assessment**: Individual churn probability for each customer
- **Top Contributing Factors**: SHAP-based feature importance analysis
- **Actionable Retention Offers**: Specific offers based on customer profile
- **Risk Level Classification**: High/Low risk categorization
- **Exportable Results**: Download recommendations as CSV

### 2. **Smart Feature Mapping**
- **IBM Dataset Features**: Contract, tenure, charges, services, etc.
- **Orange Dataset Features**: International plans, call patterns, account length, etc.
- **Contextual Recommendations**: Tailored offers based on specific risk factors

### 3. **Advanced Analytics**
- **SHAP Explanations**: Understand why customers are at risk
- **Multi-Dataset Training**: Enhanced model performance with combined data
- **Real-time Processing**: Generate recommendations on-demand

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
cd "App & Model"
pip install -r requirements_recommendations.txt
```

### Configuration
1. **Enable/Disable Features**: Edit `config.py`
   ```python
   ENABLE_RECOMMENDATIONS = True  # Set to False to disable
   ENABLE_ORANGE_DATASET = False  # Set to True if you have Orange datasets
   ```

2. **Dataset Paths**: Update paths in `config.py` if needed
   ```python
   IBM_DATA_PATH = "path/to/your/ibm_dataset.csv"
   ORANGE_DATA_PATHS = ["path/to/orange_dataset1.csv", "path/to/orange_dataset2.csv"]
   ```

### Running the App
```bash
streamlit run app.py
```

## 📊 How to Use

### 1. **Navigate to Recommendations Tab**
- The new tab appears only when recommendations are enabled
- Use sidebar filters to focus on specific customer segments

### 2. **Generate Recommendations**
- Set sample size (1-50 customers)
- Choose number of top features to analyze
- Click "Generate Recommendations" button

### 3. **Analyze Results**
- **Summary Metrics**: High-risk count, average risk score
- **Individual Analysis**: Expand each customer for detailed insights
- **Risk Factors**: Understand why customers are at risk
- **Retention Offers**: Get specific action items

### 4. **Export Results**
- Download recommendations as CSV
- Use for further analysis or CRM integration

## 🔧 Technical Details

### Architecture
```
app.py (Main App)
├── config.py (Configuration)
├── recommendation_engine.py (Core Engine)
└── app_original.py (Backup)
```

### Key Components
- **SimpleRecommendationEngine**: Main recommendation class
- **Feature Mapping**: Contextual reasons and offers
- **SHAP Integration**: Explainable AI for feature importance
- **Multi-Dataset Support**: IBM + Orange dataset compatibility

### Model Details
- **Algorithm**: LightGBM with calibration
- **Features**: Automatic encoding of categorical variables
- **Performance**: ROC-AUC typically > 0.89
- **Training**: Uses combined datasets for better generalization

## 🔄 Reverting Changes

### Quick Revert
```bash
cd "App & Model"
python revert_to_original.py
```

### Manual Revert
1. Copy `app_original.py` to `app.py`
2. Set `ENABLE_RECOMMENDATIONS = False` in `config.py`
3. Restart the app

### What Gets Reverted
- ✅ Main app functionality restored
- ✅ Original tab structure
- ✅ Original styling and layout
- ✅ Recommendation features disabled
- ✅ Configuration reset

## 📁 File Structure

```
App & Model/
├── app.py                    # Enhanced main app (with recommendations)
├── app_original.py          # Original app backup
├── config.py                # Configuration and feature flags
├── recommendation_engine.py # Recommendation engine core
├── revert_to_original.py    # Reversion script
├── requirements_recommendations.txt # New dependencies
└── README_RECOMMENDATIONS.md # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install lightgbm shap scikit-learn
   ```

2. **Feature Mismatch Errors**
   ```bash
   # Run the test script to diagnose issues
   python test_recommendations.py
   ```
   - This usually happens when training and prediction data have different features
   - The recommendation engine now handles this automatically
   - Check debug information in the recommendations tab

3. **Memory Issues**
   - Reduce sample size in recommendations
   - Disable Orange dataset integration

4. **Performance Issues**
   - Use smaller sample sizes
   - Enable caching in config

5. **Model Training Failures**
   - Check dataset paths in config.py
   - Verify data format compatibility

6. **Multiple Training Warnings**
   - The engine now uses caching to prevent retraining
   - Check if `@st.cache_resource` is working properly

### Debug Mode
Set in `config.py`:
```python
DEBUG_MODE = True
```

## 🔮 Future Enhancements

- **Real-time Scoring**: API endpoints for external systems
- **A/B Testing**: Test different retention strategies
- **Customer Segmentation**: Advanced clustering for targeted offers
- **ROI Analysis**: Measure retention campaign effectiveness
- **Integration**: CRM and marketing automation tools

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Check dataset paths and formats
4. Use the reversion script if needed

## 📝 Changelog

- **v1.0**: Initial recommendation system implementation
- **v1.1**: Added SHAP explanations and feature mapping
- **v1.2**: Multi-dataset support and configuration management
- **v1.3**: Enhanced UI and export functionality

---

**Note**: This enhancement is designed to be easily reversible. Your original app functionality is preserved in `app_original.py`.
