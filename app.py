import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Smart Elevator Monitoring System",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS FOR ENHANCED UI
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
    .alert-critical {
        background: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-warning {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-info {
        background: #dbeafe;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 5px;
    }
    .health-good { color: #16a34a; font-weight: bold; }
    .health-fair { color: #d97706; font-weight: bold; }
    .health-poor { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE INITIALIZATION
# ================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'real_time_mode' not in st.session_state:
    st.session_state.real_time_mode = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# ================================
# TITLE AND INTRODUCTION
# ================================
st.markdown("""
<div class="main-header">
    <h1>🛗 Advanced Smart Elevator Monitoring System</h1>
    <p>Real-time Predictive Maintenance | AI-Powered Anomaly Detection | Multi-Elevator Fleet Management</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🎯 Enterprise-Grade Elevator Health Management Platform
This advanced dashboard provides **real-time monitoring**, **predictive analytics**, and **intelligent maintenance scheduling** 
for elevator fleets using machine learning and statistical process control.
""")

st.divider()

# ================================
# SIDEBAR CONFIGURATION
# ================================
st.sidebar.header("⚙️ System Configuration")

# Mode Selection
mode = st.sidebar.radio(
    "Operating Mode",
    ["Dashboard Analysis", "Real-Time Monitoring", "Predictive Maintenance", "Fleet Management"],
    index=0
)

# Elevator Selection
st.sidebar.subheader("🏢 Elevator Selection")
num_elevators = st.sidebar.slider("Number of Elevators in Fleet", 1, 10, 3)
selected_elevator = st.sidebar.selectbox(
    "Select Elevator to Analyze",
    [f"Elevator {i+1}" for i in range(num_elevators)]
)

# ================================
# DATA UPLOAD SECTION
# ================================
st.sidebar.header("📂 Data Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload Elevator Dataset",
    type=["csv"],
    help="Upload CSV file with columns: ID, vibrations, humidity, revolutions, x1, x2, x3, x4, x5"
)

# Generate Sample Data Option
st.sidebar.subheader("🎲 Sample Data Generator")
if st.sidebar.button("Generate Advanced Sample Data"):
    st.session_state.data_loaded = True
    
    # Generate realistic sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Time-based features
    base_time = datetime.now() - timedelta(days=30)
    timestamps = [base_time + timedelta(minutes=i*30) for i in range(n_samples)]
    
    # Simulate realistic patterns with wear and degradation
    wear_factor = np.linspace(1.0, 1.5, n_samples)  # Increasing wear over time
    
    df = pd.DataFrame({
        'ID': range(1, n_samples + 1),
        'timestamp': timestamps,
        'elevator_id': np.random.choice([f'Elevator {i+1}' for i in range(num_elevators)], n_samples),
        
        # Sensor readings with realistic patterns
        'vibration': np.random.normal(0.5, 0.2, n_samples) * wear_factor + 
                     np.random.normal(0, 0.1, n_samples),
        'humidity': np.random.normal(45, 15, n_samples),
        'revolutions': np.random.poisson(100, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'voltage': np.random.normal(230, 10, n_samples),
        'current': np.random.normal(5, 1, n_samples),
        
        # Additional sensors
        'x1': np.random.normal(0.3, 0.1, n_samples),
        'x2': np.random.normal(0.4, 0.15, n_samples),
        'x3': np.random.normal(0.35, 0.12, n_samples),
        'x4': np.random.normal(0.45, 0.18, n_samples),
        'x5': np.random.normal(0.38, 0.14, n_samples),
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    df.loc[anomaly_indices, 'vibration'] *= 3
    df.loc[anomaly_indices, 'x1'] *= 2
    
    # Add health status
    df['health_status'] = 'Good'
    df.loc[df['vibration'] > 1.2, 'health_status'] = 'Warning'
    df.loc[df['vibration'] > 1.8, 'health_status'] = 'Critical'
    
    # Add failure probability
    df['failure_probability'] = (df['vibration'] + df['voltage'].abs() - 230) / 10
    df['failure_probability'] = np.clip(df['failure_probability'], 0, 1)
    
    st.session_state.df = df
    st.session_state.historical_data = df.copy()
    st.sidebar.success(f"✅ Generated {n_samples} samples with realistic patterns!")

# ================================
# DATA PROCESSING
# ================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Add timestamp if not present
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='30min')
    
    if 'elevator_id' not in df.columns:
        df['elevator_id'] = selected_elevator
    
    # Add health status
    df['health_status'] = 'Good'
    df.loc[df['vibration'] > df['vibration'].quantile(0.75), 'health_status'] = 'Warning'
    df.loc[df['vibration'] > df['vibration'].quantile(0.95), 'health_status'] = 'Critical'
    
    st.session_state.df = df
    st.session_state.data_loaded = True
    st.session_state.historical_data = df.copy()
    st.success("✅ Dataset Loaded Successfully!")

elif not st.session_state.data_loaded:
    st.warning("👈 Please upload a CSV file or generate sample data from the sidebar to begin.")
    st.stop()

df = st.session_state.df

# Filter by selected elevator
df_filtered = df[df['elevator_id'] == selected_elevator] if 'elevator_id' in df.columns else df

# ================================
# DASHBOARD ANALYSIS MODE
# ================================
if mode == "Dashboard Analysis":
    st.header("📊 Comprehensive Dashboard Analysis")
    
    # ========================
    # REAL-TIME KPI METRICS
    # ========================
    st.subheader("🎯 Real-Time Performance Metrics")
    
    # Calculate KPIs
    current_vibration = df_filtered['vibration'].iloc[-1] if len(df_filtered) > 0 else 0
    avg_vibration = df_filtered['vibration'].mean()
    max_vibration = df_filtered['vibration'].max()
    health_score = max(0, 100 - (current_vibration * 20))
    
    # Display KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        health_color = "🟢" if health_score > 70 else "🟡" if health_score > 40 else "🔴"
        st.metric(f"{health_color} Health Score", f"{health_score:.1f}%", 
                 delta=f"{(current_vibration - avg_vibration):.3f}")
    
    with col2:
        st.metric("Current Vibration", f"{current_vibration:.3f} mm/s",
                 delta=f"{(current_vibration - df_filtered['vibration'].iloc[-2]):.3f}" if len(df_filtered) > 1 else None)
    
    with col3:
        st.metric("Avg Revolutions", f"{df_filtered['revolutions'].mean():.1f} cycles")
    
    with col4:
        st.metric("Temperature", f"{df_filtered['temperature'].mean():.1f}°C")
    
    with col5:
        critical_count = len(df_filtered[df_filtered['health_status'] == 'Critical'])
        st.metric("Critical Alerts", critical_count, 
                 delta="High" if critical_count > 10 else "Normal")
    
    st.divider()
    
    # ========================
    # ADVANCED VISUALIZATIONS
    # ========================
    
    # 1. Multi-Variable Time Series Dashboard
    st.subheader("📈 Multi-Sensor Time Series Analysis")
    
    fig1 = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Vibration Over Time', 'Humidity Trends', 
                       'Revolutions Count', 'Temperature Variations',
                       'Voltage Fluctuations', 'Current Draw'),
        vertical_spacing=0.08
    )
    
    fig1.add_trace(
        go.Scatter(x=df_filtered.index[:500], y=df_filtered['vibration'][:500],
                   name='Vibration', line=dict(color='#ef4444')),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=df_filtered.index[:500], y=df_filtered['humidity'][:500],
                   name='Humidity', line=dict(color='#3b82f6')),
        row=1, col=2
    )
    
    fig1.add_trace(
        go.Scatter(x=df_filtered.index[:500], y=df_filtered['revolutions'][:500],
                   name='Revolutions', line=dict(color='#10b981')),
        row=2, col=1
    )
    
    if 'temperature' in df_filtered.columns:
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:500], y=df_filtered['temperature'][:500],
                       name='Temperature', line=dict(color='#f59e0b')),
            row=2, col=2
        )
    
    if 'voltage' in df_filtered.columns:
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:500], y=df_filtered['voltage'][:500],
                       name='Voltage', line=dict(color='#8b5cf6')),
            row=3, col=1
        )
    
    if 'current' in df_filtered.columns:
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:500], y=df_filtered['current'][:500],
                       name='Current', line=dict(color='#ec4899')),
            row=3, col=2
        )
    
    fig1.update_layout(height=800, showlegend=False, 
                      title_text="Real-time Multi-Sensor Monitoring Dashboard")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Sensor Correlation Matrix with Advanced Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Advanced Correlation Analysis")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_filtered[numeric_cols].corr()
        
        fig2 = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig2.update_layout(
            title="Feature Correlation Heatmap",
            width=500, height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistical Distribution Analysis")
        
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vibration Distribution', 'Humidity Distribution',
                           'Revolutions Distribution', 'Health Status Distribution')
        )
        
        fig3.add_trace(go.Histogram(x=df_filtered['vibration'], name='Vibration',
                                    marker_color='#ef4444'), row=1, col=1)
        fig3.add_trace(go.Histogram(x=df_filtered['humidity'], name='Humidity',
                                    marker_color='#3b82f6'), row=1, col=2)
        fig3.add_trace(go.Histogram(x=df_filtered['revolutions'], name='Revolutions',
                                    marker_color='#10b981'), row=2, col=1)
        
        health_counts = df_filtered['health_status'].value_counts()
        fig3.add_trace(go.Bar(x=health_counts.index, y=health_counts.values,
                             name='Health Status', marker_color=['#10b981', '#f59e0b', '#ef4444']),
                      row=2, col=2)
        
        fig3.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    # 3. Advanced Scatter Matrix
    st.subheader("🔍 Multi-Variable Relationship Analysis")
    
    features = ['vibration', 'humidity', 'revolutions']
    if 'temperature' in df_filtered.columns:
        features.append('temperature')
    
    fig4 = px.scatter_matrix(
        df_filtered[features + ['health_status']][:1000],
        dimensions=features,
        color='health_status',
        title="Multi-Dimensional Sensor Relationships",
        color_discrete_map={'Good': '#10b981', 'Warning': '#f59e0b', 'Critical': '#ef4444'}
    )
    st.plotly_chart(fig4, use_container_width=True)

# ================================
# REAL-TIME MONITORING MODE
# ================================
elif mode == "Real-Time Monitoring":
    st.header("⚡ Real-Time Monitoring Dashboard")
    
    # Real-time controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-Refresh", value=True)
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
    
    with col2:
        vibration_threshold = st.slider("🚨 Vibration Alert Threshold", 
                                      float(df_filtered['vibration'].min()),
                                      float(df_filtered['vibration'].max()),
                                      float(df_filtered['vibration'].quantile(0.90)))
    
    with col3:
        show_trends = st.checkbox("📈 Show Trend Lines", value=True)
    
    # Simulate real-time data stream
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    
    # Real-time gauge charts
    st.subheader("🎛️ Real-Time Sensor Gauges")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = df_filtered.iloc[-1]
    
    with col1:
        fig_vibration = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = latest_data['vibration'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Vibration (mm/s)"},
            delta = {'reference': df_filtered['vibration'].mean()},
            gauge = {
                'axis': {'range': [None, df_filtered['vibration'].max()]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, vibration_threshold], 'color': "lightgray"},
                    {'range': [vibration_threshold, df_filtered['vibration'].max()], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': vibration_threshold
                }
            }
        ))
        fig_vibration.update_layout(height=300)
        st.plotly_chart(fig_vibration, use_container_width=True)
    
    with col2:
        fig_temp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_data.get('temperature', 25),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Temperature (°C)"},
            gauge = {
                'axis': {'range': [0, 50]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 35], 'color': "gray"},
                    {'range': [35, 50], 'color': "red"}
                ]
            }
        ))
        fig_temp.update_layout(height=300)
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col3:
        fig_humidity = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_data['humidity'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Humidity (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        fig_humidity.update_layout(height=300)
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    with col4:
        fig_revs = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_data['revolutions'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Revolutions"},
            gauge = {
                'axis': {'range': [0, df_filtered['revolutions'].max()]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, df_filtered['revolutions'].quantile(0.5)], 'color': "lightgray"},
                    {'range': [df_filtered['revolutions'].quantile(0.5), df_filtered['revolutions'].max()], 'color': "gray"}
                ]
            }
        ))
        fig_revs.update_layout(height=300)
        st.plotly_chart(fig_revs, use_container_width=True)
    
    # Real-time alerts
    st.subheader("🚨 Real-Time Alert System")
    
    current_vibration = latest_data['vibration']
    alerts = []
    
    if current_vibration > vibration_threshold:
        alerts.append({
            'type': 'critical',
            'message': f'⚠️ CRITICAL: Vibration ({current_vibration:.3f}) exceeds threshold ({vibration_threshold:.3f})',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    if latest_data.get('temperature', 25) > 40:
        alerts.append({
            'type': 'warning',
            'message': f'🌡️ WARNING: Temperature elevated at {latest_data["temperature"]:.1f}°C',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    if latest_data['humidity'] > 80:
        alerts.append({
            'type': 'info',
            'message': f'💧 INFO: High humidity level at {latest_data["humidity"]:.1f}%',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    if alerts:
        for alert in alerts:
            alert_class = f"alert-{alert['type']}"
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{alert['timestamp']}</strong><br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-info">
            ✅ All systems operating within normal parameters
        </div>
        """, unsafe_allow_html=True)

# ================================
# PREDICTIVE MAINTENANCE MODE
# ================================
elif mode == "Predictive Maintenance":
    st.header("🔮 Predictive Maintenance & AI Analytics")
    
    # Train ML Model
    if st.button("🤖 Train Predictive Model"):
        st.session_state.model_trained = True
        
        with st.spinner("Training machine learning models..."):
            # Prepare data
            features = ['vibration', 'humidity', 'revolutions', 'x1', 'x2', 'x3', 'x4', 'x5']
            if 'temperature' in df_filtered.columns:
                features.append('temperature')
            if 'voltage' in df_filtered.columns:
                features.append('voltage')
            
            X = df_filtered[features].dropna()
            y = (df_filtered.loc[X.index, 'vibration'] > df_filtered['vibration'].quantile(0.75)).astype(int)
            
            # Train Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Train Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X)
            
            # Store models in session state
            st.session_state.rf_model = rf_model
            st.session_state.iso_forest = iso_forest
            st.session_state.features = features
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.session_state.feature_importance = feature_importance
            
        st.success("✅ ML Models trained successfully!")
    
    # Predictive Analytics Section
    if st.session_state.model_trained:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Feature Importance Analysis")
            
            fig_importance = px.bar(
                st.session_state.feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Key Factors Influencing Elevator Health",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Anomaly Detection Results")
            
            # Detect anomalies
            X = df_filtered[st.session_state.features].dropna()
            anomalies = st.session_state.iso_forest.predict(X)
            anomaly_scores = st.session_state.iso_forest.decision_function(X)
            
            anomaly_df = pd.DataFrame({
                'ID': X.index,
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
                color_discrete_map={True: 'red', False: 'blue'}
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Maintenance Predictions
        st.subheader("🔧 Predictive Maintenance Schedule")
        
        # Predict failures
        latest_features = df_filtered[st.session_state.features].iloc[-1:].values
        failure_prob = st.session_state.rf_model.predict_proba(latest_features)[0][1]
        
        st.info(f"""
        **Failure Probability Analysis for {selected_elevator}**
        
        - Current Failure Probability: **{failure_prob*100:.2f}%**
        - Recommended Action: {'🚨 Immediate Maintenance Required' if failure_prob > 0.7 else 
                             '⚠️ Schedule Maintenance Within Week' if failure_prob > 0.4 else 
                             '✅ Continue Regular Monitoring'}
        """)
        
        # Component Health Scores
        st.subheader("🏥 Component Health Assessment")
        
        components = {
            'Door System': df_filtered['vibration'].mean() / df_filtered['vibration'].max(),
            'Motor': df_filtered['revolutions'].mean() / df_filtered['revolutions'].max(),
            'Environment': df_filtered['humidity'].mean() / 100,
        }
        
        if 'temperature' in df_filtered.columns:
            components['Control System'] = df_filtered['temperature'].mean() / 50
        
        for component, score in components.items():
            health_percentage = (1 - score) * 100
            health_class = "health-good" if health_percentage > 70 else "health-fair" if health_percentage > 40 else "health-poor"
            
            st.markdown(f"""
            <div style="margin: 10px 0; padding: 15px; border-radius: 5px; background: #f8f9fa;">
                <strong>📋 {component}</strong>
                <div style="margin-top: 10px;">
                    <progress value="{health_percentage}" max="100" style="width: 100%; height: 20px;"></progress>
                </div>
                <p class="{health_class}">Health Score: {health_percentage:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cost Estimation
        st.subheader("💰 Maintenance Cost Estimation")
        
        estimated_downtime = failure_prob * 24  # hours
        cost_per_hour = 500  # hypothetical cost
        estimated_cost = estimated_downtime * cost_per_hour
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Potential Downtime", f"{estimated_downtime:.1f} hours")
        
        with col2:
            st.metric("Estimated Revenue Loss", f"${estimated_cost:,.2f}")
        
        with col3:
            st.metric("Recommended Maintenance", 
                     "Immediate" if failure_prob > 0.7 else "Scheduled" if failure_prob > 0.4 else "Routine")

# ================================
# FLEET MANAGEMENT MODE
# ================================
elif mode == "Fleet Management":
    st.header("🏢 Multi-Elevator Fleet Management")
    
    # Fleet Overview
    st.subheader("📊 Fleet Performance Overview")
    
    elevator_stats = []
    for elevator in [f'Elevator {i+1}' for i in range(num_elevators)]:
        if elevator in df['elevator_id'].values:
            elevator_data = df[df['elevator_id'] == elevator]
            stats = {
                'Elevator': elevator,
                'Health Score': max(0, 100 - elevator_data['vibration'].mean() * 20),
                'Avg Vibration': elevator_data['vibration'].mean(),
                'Total Cycles': elevator_data['revolutions'].sum(),
                'Critical Alerts': len(elevator_data[elevator_data['health_status'] == 'Critical']),
                'Last Maintenance': '2024-01-15'  # Placeholder
            }
            elevator_stats.append(stats)
    
    fleet_df = pd.DataFrame(elevator_stats)
    
    # Display fleet table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            fleet_df,
            column_config={
                'Health Score': st.column_config.ProgressColumn(
                    'Health Score',
                    help='Overall elevator health percentage',
                    format='%.1f%%',
                    min_value=0,
                    max_value=100
                ),
                'Critical Alerts': st.column_config.NumberColumn(
                    'Critical Alerts',
                    help='Number of critical alerts',
                    format='%d'
                )
            }
        )
    
    with col2:
        st.subheader("🏆 Fleet Health Distribution")
        
        health_categories = pd.cut(fleet_df['Health Score'], 
                                  bins=[0, 40, 70, 100],
                                  labels=['Critical', 'Warning', 'Good'])
        
        fig_fleet = px.pie(
            health_categories.value_counts(),
            names=health_categories.value_counts().index,
            title='Fleet Health Status Distribution',
            color_discrete_map={'Critical': '#ef4444', 'Warning': '#f59e0b', 'Good': '#10b981'}
        )
        st.plotly_chart(fig_fleet, use_container_width=True)
    
    # Individual Elevator Comparison
    st.subheader("🔄 Elevator Performance Comparison")
    
    fig_comparison = go.Figure()
    
    for i, elevator in enumerate(fleet_df['Elevator']):
        fig_comparison.add_trace(go.Scatter(
            x=['Vibration', 'Cycles', 'Alerts'],
            y=[
                fleet_df[fleet_df['Elevator'] == elevator]['Avg Vibration'].values[0],
                fleet_df[fleet_df['Elevator'] == elevator]['Total Cycles'].values[0] / 1000,
                fleet_df[fleet_df['Elevator'] == elevator]['Critical Alerts'].values[0]
            ],
            mode='lines+markers',
            name=elevator,
            line=dict(width=2)
        ))
    
    fig_comparison.update_layout(
        title='Performance Metrics Comparison Across Fleet',
        xaxis_title='Metrics',
        yaxis_title='Normalized Values',
        hovermode='x unified'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Fleet Maintenance Schedule
    st.subheader("📅 Fleet Maintenance Schedule")
    
    maintenance_schedule = pd.DataFrame({
        'Elevator': fleet_df['Elevator'],
        'Priority': ['High' if score < 40 else 'Medium' if score < 70 else 'Low' 
                    for score in fleet_df['Health Score']],
        'Recommended Action': ['Immediate Inspection' if score < 40 else 
                              'Schedule Within Week' if score < 70 else 
                              'Routine Check' for score in fleet_df['Health Score']],
        'Est. Cost': ['$2,500' if score < 40 else '$800' if score < 70 else '$200' 
                     for score in fleet_df['Health Score']],
        'Lead Time': ['24 hours' if score < 40 else '1 week' if score < 70 else '1 month' 
                     for score in fleet_df['Health Score']]
    })
    
    st.dataframe(maintenance_schedule)

# ================================
# EXPORT FUNCTIONALITY
# ================================
st.divider()
st.header("📤 Export & Reporting")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Analysis Report"):
        report_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Elevator': selected_elevator,
            'Total Samples': len(df_filtered),
            'Avg Vibration': df_filtered['vibration'].mean(),
            'Max Vibration': df_filtered['vibration'].max(),
            'Health Status': df_filtered['health_status'].mode()[0],
            'Critical Alerts': len(df_filtered[df_filtered['health_status'] == 'Critical'])
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="Download Report (CSV)",
            data=csv,
            file_name=f"elevator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📥 Export Full Dataset"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=csv,
            file_name=f"elevator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    st.info("💡 Additional export formats (PDF, Excel) available in premium version")

# ================================
# FOOTER
# ================================
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
    <h4>🛗 Smart Elevator Monitoring System v2.0</h4>
    <p>Advanced Predictive Maintenance Platform | Real-Time Analytics | AI-Powered Insights</p>
    <p style="font-size: 0.8em; color: #666;">
        © 2024 Smart Building Solutions | Built with Streamlit, Machine Learning & Advanced Analytics
    </p>
</div>
""", unsafe_allow_html=True)
