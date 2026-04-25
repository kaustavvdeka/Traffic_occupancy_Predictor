import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image

try:
    from parking_model import ParkingPredictor, DataPreprocessor, VehicleDetector
    import __main__
    
    # Pickle expects these exact classes to be located in __main__ because
    # they were pickled when parking_model.py was executed directly as main script.
    __main__.DataPreprocessor = DataPreprocessor
    __main__.VehicleDetector = VehicleDetector
except ImportError:
    st.error("Could not import ParkingPredictor. Make sure parking_model.py is in the same directory.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Parking ML Model Evaluation", page_icon="🚘", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_predictor():
    model_path = "parking_model.pkl"
    if not os.path.exists(model_path):
        return None
    predictor = ParkingPredictor()
    predictor.load_model(model_path)
    return predictor

st.title("🚘 Smart Parking Prediction")
st.subheader("Machine Learning Model Evaluation Dashboard")

predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ `parking_model.pkl` not found. Please train the model first by running `python3 parking_model.py`.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Evaluation Metrics", "🔑 Feature Importance", "🧪 Live Testing"])

# --- Tab 1: Evaluation Metrics ---
with tab1:
    st.header("Model Performance Statistics")
    
    stats = predictor.training_stats
    if not stats:
        st.info("No training stats were found in the model pickle. Re-run training to generate them.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Testing Set")
            st.metric(label="Mean Absolute Error (MAE)", value=f"{stats.get('test_mae', 0):.2f}%")
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{stats.get('test_rmse', 0):.2f}%")
            st.metric(label="R-Squared (R²)", value=f"{stats.get('test_r2', 0):.4f}")
            st.metric(label="Explained Variance", value=f"{stats.get('test_explained_var', 0):.4f}")
        
        with col2:
            st.markdown("### 📚 Training Set")
            st.metric(label="Train MAE", value=f"{stats.get('train_mae', 0):.2f}%")
            st.metric(label="Train RMSE", value=f"{stats.get('train_rmse', 0):.2f}%")
            st.metric(label="Train R²", value=f"{stats.get('train_r2', 0):.4f}")
            cross_val_mae = stats.get('cv_mae_mean', 0)
            cross_val_std = stats.get('cv_mae_std', 0)
            if cross_val_mae > 0:
                st.metric(label="5-Fold CV MAE Mean", value=f"{cross_val_mae:.2f}% ± {cross_val_std:.2f}%")
        
        st.divider()
        st.markdown("### 📈 Visual Result Plots")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("test_results.png"):
                st.image("test_results.png", caption="Test Dataset Actual vs Predicted", use_column_width=True)
            else:
                st.info("No `test_results.png` found.")
        with c2:
            if os.path.exists("training_results.png"):
                st.image("training_results.png", caption="Training Dataset Metrics", use_column_width=True)
            else:
                st.info("No `training_results.png` found.")

# --- Tab 2: Feature Importance ---
with tab2:
    st.header("Feature Importance")
    st.write("Understand which variables the Gradient Boosting model relies on the most.")
    
    if hasattr(predictor.model, 'feature_importances_') and hasattr(predictor.preprocessor, 'feature_columns'):
        importance_df = pd.DataFrame({
            'Feature': predictor.preprocessor.feature_columns,
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'].head(15)[::-1], importance_df['Importance'].head(15)[::-1], color='mediumpurple')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top Features used by RandomForestRegressor')
        st.pyplot(fig)
        
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("The loaded model lacks feature importance attributes.")

# --- Tab 3: Live Testing ---
with tab3:
    st.header("Live Image Prediction")
    st.markdown("Upload a parking lot overview image, and the model will extract computer vision features to predict occupancy percentages!")
    
    uploaded_file = st.file_uploader("Upload Image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Parking Image", use_column_width=True)
            
            # Save uploaded image to temp file for CV2 to process
            temp_path = "temp_uploaded_img.jpg"
            image.save(temp_path)
            
            st.info("Extracting computer vision features using VehicleDetector pipeline...")
            result = predictor.predict_from_image(temp_path)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Analysis Complete!")
                colA, colB, colC = st.columns(3)
                colA.metric("Estimated Base Capacity", f"{result.get('detected_occupancy', 0):.1f}% Raw Base")
                colB.metric("Detected Vehicles", f"{result.get('detected_vehicles', 0)}")
                
                # The final pipeline output
                final_pred = result.get('predicted_occupancy', 0)
                colC.metric("🤖 ML PREDICTED OCCUPANCY", f"{final_pred:.1f}%", 
                            delta="High Congestion" if final_pred > 80 else "Available", 
                            delta_color="inverse" if final_pred > 80 else "normal")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
