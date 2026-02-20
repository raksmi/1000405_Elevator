import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Smart Elevator Monitoring Dashboard",
    page_icon="🛗",
    layout="wide"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .alert-warning {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# TITLE AND INTRODUCTION
# ================================
st.markdown("""
<div class="main-header">
    <h1>🛗 Smart Elevator Monitoring Dashboard</h1>
    <p>Predictive Maintenance & Sensor-Based Elevator Monitoring</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This dashboard analyzes elevator sensor readings including **humidity, revolutions, and vibration**, 
supporting smarter maintenance decisions.
""")

st.divider()

# ================================
# FILE UPLOAD
# ================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Elevator CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload your elevator dataset CSV file to continue.")
    st.stop()

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(uploaded_file)

st.success("✅ Dataset Uploaded Successfully!")

# ================================
# DATA VALIDATION
# ================================
required_columns = ['ID', 'revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']

# Check if all required columns are present
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# ================================
# MODE SELECTION
# ================================
st.sidebar.header("🎯 Dashboard Mode")

mode = st.sidebar.radio(
    "Select Mode",
    ["Dashboard Analysis", "Predictive Maintenance"],
    index=0
)

# ================================
# COMMON: DATA CLEANING
# ================================
# Remove duplicates
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

# Check for missing values
missing_values = df.isnull().sum().sum()

if missing_values > 0:
    st.warning(f"Dataset contains {missing_values} missing values. Removing rows with missing data.")
    df = df.dropna()

# Convert numeric columns safely
numeric_cols = required_columns[1:]  # All columns except ID
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# ================================
# DASHBOARD ANALYSIS MODE
# ================================
if mode == "Dashboard Analysis":
    st.header("📊 Dashboard Analysis")
    
    # ========================
    # KEY METRICS
    # ========================
    st.subheader("📌 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_vibration = df['vibration'].mean()
        st.metric("Average Vibration", f"{avg_vibration:.3f}")
    
    with col2:
        max_vibration = df['vibration'].max()
        st.metric("Maximum Vibration", f"{max_vibration:.3f}")
    
    with col3:
        avg_humidity = df['humidity'].mean()
        st.metric("Average Humidity", f"{avg_humidity:.2f}")
    
    with col4:
        total_revolutions = df['revolutions'].sum()
        st.metric("Total Revolutions", f"{total_revolutions:,.0f}")
    
    st.divider()
    
    # ========================
    # VIBRATION THRESHOLD SLIDER
    # ========================
    st.subheader("⚙️ Vibration Threshold Filter")
    
    vibration_threshold = st.slider(
        "Set Vibration Alert Threshold",
        float(df['vibration'].min()),
        float(df['vibration'].max()),
        float(df['vibration'].quantile(0.75))
    )
    
    # Filter data based on threshold
    high_vibration_data = df[df['vibration'] > vibration_threshold]
    
    st.write(f"Samples with vibration above **{vibration_threshold:.3f}**: {len(high_vibration_data)}")
    
    if len(high_vibration_data) > 0:
        st.markdown(f"""
        <div class="alert-warning">
            <strong>⚠️ High Vibration Alert:</strong> {len(high_vibration_data)} samples exceed the threshold.
            These may indicate mechanical issues requiring attention.
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================
    # REQUIRED VISUALIZATIONS
    # ========================
    st.header("📈 Exploratory Data Analysis Visualizations")
    
    # 1. LINE PLOT (ID vs Vibration)
    st.subheader("1️⃣ Vibration Over Time (Line Plot)")
    
    fig1 = px.line(
        df,
        x="ID",
        y="vibration",
        title="Vibration Trend Over Time (ID vs Vibration)"
    )
    
    # Add threshold line
    fig1.add_hline(
        y=vibration_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {vibration_threshold:.3f}"
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. HISTOGRAM (Humidity Distribution)
    st.subheader("2️⃣ Humidity Distribution (Histogram)")
    
    fig2 = px.histogram(
        df,
        x="humidity",
        title="Humidity Distribution",
        nbins=50,
        color_discrete_sequence=['#3b82f6']
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. HISTOGRAM (Revolutions Distribution)
    st.subheader("3️⃣ Revolutions Distribution (Histogram)")
    
    fig3 = px.histogram(
        df,
        x="revolutions",
        title="Revolutions Distribution",
        nbins=50,
        color_discrete_sequence=['#10b981']
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. SCATTER PLOT (Revolutions vs Vibration)
    st.subheader("4️⃣ Revolutions vs Vibration (Scatter Plot)")
    
    fig4 = px.scatter(
        df,
        x="revolutions",
        y="vibration",
        trendline="ols",
        title="Revolutions vs Vibration Relationship",
        color_discrete_sequence=['#8b5cf6']
    )
    
    # Add threshold line
    fig4.add_hline(
        y=vibration_threshold,
        line_dash="dash",
        line_color="red"
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # 5. BOX PLOT (x1, x2, x3, x4, x5)
    st.subheader("5️⃣ Sensor Variability & Outliers (Box Plot)")
    
    sensor_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
    sensor_data = df[sensor_cols].melt(var_name='Sensor', value_name='Value')
    
    fig5 = px.box(
        sensor_data,
        x='Sensor',
        y='Value',
        title="Box Plot of Sensor Readings (x1-x5)",
        color='Sensor'
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # 6. CORRELATION HEATMAP
    st.subheader("6️⃣ Correlation Heatmap (All Numeric Columns)")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    fig6 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig6.update_layout(
        title="Correlation Heatmap of All Numeric Features",
        width=800,
        height=700
    )
    st.plotly_chart(fig6, use_container_width=True)

# ================================
# PREDICTIVE MAINTENANCE MODE
# ================================
elif mode == "Predictive Maintenance":
    st.header("🔮 Predictive Maintenance")
    
    # Train ML Model Button
    if st.button("🤖 Train Predictive Models"):
        st.session_state.model_trained = True
        
        with st.spinner("Training machine learning models..."):
            # Prepare features
            features = ['revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']
            X = df[features]
            
            # Create target: High vibration (above 75th percentile)
            threshold_75th = df['vibration'].quantile(0.75)
            y = (df['vibration'] > threshold_75th).astype(int)
            
            # Train RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Train IsolationForest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X)
            
            # Store models in session state
            st.session_state.rf_model = rf_model
            st.session_state.iso_forest = iso_forest
            st.session_state.features = features
            st.session_state.threshold_75th = threshold_75th
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.session_state.feature_importance = feature_importance
            
        st.success("✅ ML Models trained successfully!")
        st.info(f"High vibration threshold (75th percentile): {threshold_75th:.3f}")
    
    # Display results if model is trained
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Feature Importance Analysis")
            
            fig_importance = px.bar(
                st.session_state.feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Key Factors Influencing High Vibration",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Anomaly Detection Results")
            
            # Detect anomalies
            X = df[st.session_state.features]
            anomalies = st.session_state.iso_forest.predict(X)
            anomaly_scores = st.session_state.iso_forest.decision_function(X)
            
            # Create anomaly dataframe
            anomaly_df = pd.DataFrame({
                'ID': df['ID'],
                'Anomaly_Score': anomaly_scores,
                'Is_Anomaly': anomalies == -1
            })
            
            anomaly_count = anomaly_df['Is_Anomaly'].sum()
            
            fig_anomaly = px.scatter(
                anomaly_df,
                x='ID',
                y='Anomaly_Score',
                color='Is_Anomaly',
                title=f"Anomaly Detection Results ({anomaly_count} anomalies detected)",
                color_discrete_map={True: 'red', False: 'blue'},
                labels={'Is_Anomaly': 'Anomaly', 'Anomaly_Score': 'Anomaly Score'}
            )
            
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            st.info(f"Anomaly Detection: {anomaly_count} anomalies found out of {len(df)} samples")
        
        # Display anomalous samples
        st.subheader("📋 Anomalous Samples")
        
        anomalous_samples = df.iloc[anomaly_df[anomaly_df['Is_Anomaly']].index]
        
        if len(anomalous_samples) > 0:
            st.dataframe(anomalous_samples, use_container_width=True)
        else:
            st.info("No anomalous samples detected.")
    
    else:
        st.info("👈 Click 'Train Predictive Models' to begin analysis.")

# ================================
# DATA SUMMARY
# ================================
st.divider()
st.header("📋 Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### Dataset Info")
    st.write(f"**Rows:** {len(df)}")
    st.write(f"**Columns:** {len(df.columns)}")
    st.write(f"**Duplicates Removed:** {duplicates}")

with col2:
    st.write("### First 5 Rows")
    st.dataframe(df.head())

with col3:
    st.write("### Column Names")
    st.write(df.columns.tolist())

# ================================
# FOOTER
# ================================
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
    <h4>🛗 Smart Elevator Monitoring System</h4>
    <p>Predictive Maintenance & Sensor-Based Elevator Monitoring</p>
</div>
""", unsafe_allow_html=True)
