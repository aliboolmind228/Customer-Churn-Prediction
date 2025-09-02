#!/usr/bin/env python3
"""
Revert App to Original State
============================

This script reverts the app.py file back to its original state
by copying app_original.py back to app.py.

Usage:
    python revert_to_original.py

To re-enable recommendations later:
    1. Set ENABLE_RECOMMENDATIONS = True in config.py
    2. Restart the Streamlit app
"""

import shutil
import os
import sys

def revert_app():
    """Revert app.py to original state"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file = os.path.join(current_dir, "app_original.py")
    current_file = os.path.join(current_dir, "app.py")
    
    if not os.path.exists(original_file):
        print("❌ Error: app_original.py not found!")
        print("Cannot revert without the original backup file.")
        return False
    
    try:
        # Create backup of current modified file
        backup_file = os.path.join(current_dir, "app_modified_backup.py")
        shutil.copy2(current_file, backup_file)
        print(f"✅ Created backup of current modified file: {backup_file}")
        
        # Copy original back to app.py
        shutil.copy2(original_file, current_file)
        print("✅ Successfully reverted app.py to original state!")
        
        # Also disable recommendations in config
        config_file = os.path.join(current_dir, "config.py")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Replace ENABLE_RECOMMENDATIONS = True with False
            config_content = config_content.replace(
                "ENABLE_RECOMMENDATIONS = True",
                "ENABLE_RECOMMENDATIONS = False"
            )
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            print("✅ Disabled recommendations in config.py")
        
        print("\n🔄 Reversion complete!")
        print("📝 To re-enable recommendations later:")
        print("   1. Set ENABLE_RECOMMENDATIONS = True in config.py")
        print("   2. Restart the Streamlit app")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during reversion: {e}")
        return False

def main():
    """Main function"""
    print("🔄 Customer Churn Dashboard - Reversion Tool")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: Please run this script from the 'App & Model' directory")
        sys.exit(1)
    
    # Confirm reversion
    print("⚠️  This will revert your app.py to its original state.")
    print("📁 A backup of your current modified version will be saved.")
    
    confirm = input("\n❓ Are you sure you want to continue? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        if revert_app():
            print("\n🎉 Reversion completed successfully!")
        else:
            print("\n💥 Reversion failed!")
            sys.exit(1)
    else:
        print("\n❌ Reversion cancelled.")

if __name__ == "__main__":
    main()
