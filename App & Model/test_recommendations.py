#!/usr/bin/env python3
"""
Test Recommendations Engine
==========================

This script tests the recommendation engine to identify any issues
before running it in the main Streamlit app.

Usage:
    python test_recommendations.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import SimpleRecommendationEngine, load_combined_datasets, encode_categorical_variables
from config import *

def test_recommendation_engine():
    """Test the recommendation engine step by step"""
    
    print("🧪 Testing Recommendation Engine")
    print("=" * 50)
    
    try:
        # Step 1: Load datasets
        print("\n1️⃣ Loading datasets...")
        combined_data = load_combined_datasets(IBM_DATA_PATH)
        if combined_data is None:
            print("❌ Failed to load datasets")
            return False
        
        print(f"✅ Dataset loaded: {combined_data.shape}")
        print(f"   Columns: {combined_data.columns.tolist()}")
        
        # Step 2: Encode categorical variables
        print("\n2️⃣ Encoding categorical variables...")
        encoded_data = encode_categorical_variables(combined_data)
        print(f"✅ Data encoded: {encoded_data.shape}")
        print(f"   Data types: {encoded_data.dtypes.value_counts().to_dict()}")
        
        # Step 3: Initialize and train model
        print("\n3️⃣ Training recommendation engine...")
        engine = SimpleRecommendationEngine()
        X_test, y_test = engine.train_model(encoded_data)
        print(f"✅ Model trained successfully")
        print(f"   Training features: {len(engine.feature_names)}")
        print(f"   Test set size: {len(X_test)}")
        
        # Step 4: Test data preparation
        print("\n4️⃣ Testing data preparation...")
        sample_data = encoded_data.head(5)
        
        # Check data types before preparation
        print(f"   Sample data types: {sample_data.dtypes.value_counts().to_dict()}")
        
        prepared_data = engine.prepare_data_for_prediction(sample_data)
        print(f"✅ Data preparation successful")
        print(f"   Original features: {len(sample_data.columns)}")
        print(f"   Prepared features: {len(prepared_data.columns)}")
        print(f"   Feature match: {'✅' if len(prepared_data.columns) == len(engine.feature_names) else '❌'}")
        print(f"   Prepared data types: {prepared_data.dtypes.value_counts().to_dict()}")
        
        # Step 5: Test recommendations
        print("\n5️⃣ Testing recommendations...")
        recommendations = engine.get_recommendations(sample_data, top_k=3)
        print(f"✅ Recommendations generated successfully")
        print(f"   Number of recommendations: {len(recommendations)}")
        print(f"   Columns: {recommendations.columns.tolist()}")
        
        # Step 6: Display sample recommendation
        print("\n6️⃣ Sample recommendation:")
        if len(recommendations) > 0:
            sample_rec = recommendations.iloc[0]
            print(f"   Prediction: {sample_rec['prediction']}")
            print(f"   Probability: {sample_rec['prediction_proba']:.3f}")
            print(f"   Top Features: {sample_rec['top_features'][:3]}")
            print(f"   Risk Factors: {sample_rec['reasons']}")
            print(f"   Retention Offers: {sample_rec['retention_offers'][:2]}")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_mapping():
    """Test the feature mapping functionality"""
    
    print("\n🔍 Testing Feature Mapping")
    print("=" * 30)
    
    try:
        from recommendation_engine import create_feature_mapping
        
        mapping = create_feature_mapping()
        print(f"✅ Feature mapping created: {len(mapping)} features")
        
        # Test a few features
        test_features = ['Contract', 'tenure', 'MonthlyCharges']
        for feature in test_features:
            if feature in mapping:
                print(f"   {feature}: {len(mapping[feature]['reasons'])} reasons, {len(mapping[feature]['offers'])} offers")
            else:
                print(f"   {feature}: ❌ Not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature mapping test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Recommendation Engine Test Suite")
    print("=" * 50)
    
    # Test feature mapping
    if not test_feature_mapping():
        print("\n❌ Feature mapping tests failed!")
        return
    
    # Test recommendation engine
    if not test_recommendation_engine():
        print("\n❌ Recommendation engine tests failed!")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("\n💡 The recommendation engine is ready to use in the Streamlit app.")

if __name__ == "__main__":
    main()
