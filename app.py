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
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os
import base64
import io
import json
import random
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="🛗 Ultimate Smart Elevator Monitoring System",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS - MIND-BLOWING DESIGN
# ================================
def get_theme_colors():
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    
    if st.session_state.theme_mode == 'dark':
        return {
            'bg': '#0f172a',
            'card_bg': '#1e293b',
            'text': '#f8fafc',
            'accent': '#8b5cf6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'border': '#334155',
            'gradient1': '#667eea',
            'gradient2': '#764ba2'
        }
    else:
        return {
            'bg': '#f8fafc',
            'card_bg': '#ffffff',
            'text': '#1e293b',
            'accent': '#8b5cf6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'border': '#e2e8f0',
            'gradient1': '#667eea',
            'gradient2': '#764ba2'
        }

colors = get_theme_colors()

st.markdown(f"""
<style>
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {colors['bg']} 0%, {colors['card_bg']} 100%);
    }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: {colors['bg']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {colors['accent']};
        border-radius: 5px;
    }}
    
    /* Animated Header */
    .main-header {{
        background: linear-gradient(135deg, {colors['gradient1']} 0%, {colors['gradient2']} 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}
    
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: {colors['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border: 1px solid {colors['border']};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, {colors['gradient1']}, {colors['gradient2']});
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }}
    
    /* Alert Boxes */
    .alert-critical {{
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #dc2626;
        padding: 1.2rem;
        border-radius: 10px;
        animation: pulse 2s infinite;
    }}
    
    .alert-warning {{
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #d97706;
        padding: 1.2rem;
        border-radius: 10px;
    }}
    
    .alert-info {{
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #2563eb;
        padding: 1.2rem;
        border-radius: 10px;
    }}
    
    .alert-success {{
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1.2rem;
        border-radius: 10px;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    /* Health Indicators */
    .health-good {{ color: {colors['success']}; font-weight: bold; }}
    .health-fair {{ color: {colors['warning']}; font-weight: bold; }}
    .health-poor {{ color: {colors['danger']}; font-weight: bold; }}
    
    /* Progress Bars */
    .progress-container {{
        background: {colors['border']};
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }}
    
    .progress-bar {{
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        animation: shimmer 2s infinite;
        background: linear-gradient(90deg, {colors['gradient1']}, {colors['gradient2']}, {colors['gradient1']});
        background-size: 200% 100%;
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    /* Logo Container */
    .logo-container {{
        text-align: center;
        padding: 2rem;
        background: {colors['card_bg']};
        border-radius: 20px;
        margin-top: 3rem;
        border: 2px solid {colors['border']};
    }}
    
    .logo-container img {{
        max-width: 150px;
        animation: float 3s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    /* Notification Badge */
    .notification-badge {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: {colors['danger']};
        color: white;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
        animation: bounce 1s infinite;
        z-index: 1000;
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
    }}
    
    /* Stats Cards Animation */
    .stats-card {{
        background: {colors['card_bg']};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors['border']};
        position: relative;
        overflow: hidden;
    }}
    
    .stats-card::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, {colors['accent']}22 0%, transparent 70%);
    }}
    
    /* Clock Widget */
    .clock-widget {{
        background: {colors['card_bg']};
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid {colors['accent']};
        font-size: 1.2rem;
        font-weight: bold;
        color: {colors['text']};
    }}
    
    /* Weather Widget */
    .weather-widget {{
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }}
    
    /* Achievement Badge */
    .achievement-badge {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
        animation: pop 0.5s ease;
    }}
    
    @keyframes pop {{
        0% {{ transform: scale(0); }}
        80% {{ transform: scale(1.2); }}
        100% {{ transform: scale(1); }}
    }}
    
    /* Custom Button */
    .custom-button {{
        background: linear-gradient(135deg, {colors['gradient1']}, {colors['gradient2']});
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .custom-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }}
    
    /* Chart Container */
    .chart-container {{
        background: {colors['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid {colors['border']};
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
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
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': 'Admin User',
        'role': 'System Administrator',
        'level': 5,
        'experience': 1500
    }
if 'system_health' not in st.session_state:
    st.session_state.system_health = {
        'cpu_usage': 45,
        'memory_usage': 62,
        'disk_usage': 38,
        'network_latency': 23
    }

# ================================
# UTILITY FUNCTIONS
# ================================

def auto_load_csv():
    """Automatically detect and load CSV files from common locations"""
    possible_paths = [
        'elevator_data.csv',
        'data/elevator_data.csv',
        'dataset.csv',
        'elevators.csv',
        'sensor_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except:
                continue
    return None

def create_sample_data():
    """Create comprehensive sample data with realistic patterns"""
    np.random.seed(42)
    n_samples = 15000
    
    # Time-based features
    base_time = datetime.now() - timedelta(days=45)
    timestamps = [base_time + timedelta(minutes=i*15) for i in range(n_samples)]
    
    # Elevator IDs
    elevator_ids = [f'Elevator {i+1}' for i in range(8)]
    
    # Simulate realistic patterns with wear and degradation
    time_factor = np.linspace(1.0, 1.6, n_samples)  # Increasing wear over time
    
    # Create patterns
    daily_pattern = np.sin(np.arange(n_samples) * 2 * np.pi / 96)  # 24h pattern
    weekly_pattern = np.sin(np.arange(n_samples) * 2 * np.pi / 672)  # 7d pattern
    
    df = pd.DataFrame({
        'ID': range(1, n_samples + 1),
        'timestamp': timestamps,
        'elevator_id': np.random.choice(elevator_ids, n_samples),
        'floor': np.random.randint(1, 51, n_samples),
        'direction': np.random.choice(['Up', 'Down', 'Idle'], n_samples),
        
        # Sensor readings with realistic patterns
        'vibration': (np.random.normal(0.5, 0.2, n_samples) * time_factor + 
                     np.random.normal(0, 0.1, n_samples) * daily_pattern +
                     np.random.normal(0, 0.05, n_samples) * weekly_pattern),
        'humidity': np.random.normal(45, 15, n_samples) + 5 * daily_pattern,
        'revolutions': np.random.poisson(100, n_samples) + 10 * daily_pattern.astype(int),
        'temperature': np.random.normal(25, 5, n_samples) + 2 * daily_pattern,
        'voltage': np.random.normal(230, 10, n_samples),
        'current': np.random.normal(5, 1, n_samples),
        'pressure': np.random.normal(101.3, 2, n_samples),
        
        # Additional sensors
        'x1': np.random.normal(0.3, 0.1, n_samples),
        'x2': np.random.normal(0.4, 0.15, n_samples),
        'x3': np.random.normal(0.35, 0.12, n_samples),
        'x4': np.random.normal(0.45, 0.18, n_samples),
        'x5': np.random.normal(0.38, 0.14, n_samples),
        
        # Energy metrics
        'energy_consumption': np.random.normal(2.5, 0.5, n_samples),
        'power_factor': np.random.normal(0.9, 0.05, n_samples),
        
        # Performance metrics
        'travel_time': np.random.normal(45, 10, n_samples),
        'waiting_time': np.random.normal(30, 8, n_samples),
        'door_cycles': np.random.poisson(80, n_samples),
    })
    
    # Add anomalies
    anomaly_indices = np.random.choice(n_samples, int(n_samples * 0.03), replace=False)
    df.loc[anomaly_indices, 'vibration'] *= 2.5
    df.loc[anomaly_indices, 'temperature'] += 5
    df.loc[anomaly_indices, 'x1'] *= 2
    
    # Add health status
    df['health_status'] = 'Good'
    df.loc[df['vibration'] > 1.0, 'health_status'] = 'Warning'
    df.loc[df['vibration'] > 1.5, 'health_status'] = 'Critical'
    
    # Add failure probability
    df['failure_probability'] = (df['vibration'] + 
                                 abs(df['voltage'] - 230) / 50 + 
                                 (df['temperature'] - 25) / 50)
    df['failure_probability'] = np.clip(df['failure_probability'], 0, 1)
    
    # Add maintenance indicators
    df['days_since_maintenance'] = np.random.randint(1, 180, n_samples)
    df['maintenance_cost'] = np.random.exponential(500, n_samples)
    
    return df

def calculate_health_score(vibration, temperature, humidity):
    """Calculate comprehensive health score"""
    base_score = 100
    
    # Deduct points for each factor
    vibration_penalty = (vibration - 0.5) * 20
    temperature_penalty = max(0, abs(temperature - 25) - 5) * 2
    humidity_penalty = max(0, abs(humidity - 50) - 20) * 0.5
    
    health_score = base_score - vibration_penalty - temperature_penalty - humidity_penalty
    return max(0, min(100, health_score))

def add_notification(message, level='info'):
    """Add notification to the system"""
    notification = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'message': message,
        'level': level
    }
    st.session_state.notifications.append(notification)
    if len(st.session_state.notifications) > 10:
        st.session_state.notifications.pop(0)

def add_achievement(achievement):
    """Add achievement to user profile"""
    if achievement not in st.session_state.achievements:
        st.session_state.achievements.append(achievement)
        add_notification(f"🏆 Achievement Unlocked: {achievement}", 'success')

# ================================
# AUTO LOAD DATA
# ================================
if not st.session_state.data_loaded:
    # Try to auto-load CSV
    auto_df = auto_load_csv()
    
    if auto_df is not None:
        df = auto_df
        add_notification("✅ CSV file automatically detected and loaded!", 'success')
    else:
        # Generate sample data
        df = create_sample_data()
        add_notification("📊 Sample data generated for demonstration", 'info')
    
    # Add health status if not present
    if 'health_status' not in df.columns:
        df['health_status'] = 'Good'
        df.loc[df['vibration'] > df['vibration'].quantile(0.75), 'health_status'] = 'Warning'
        df.loc[df['vibration'] > df['vibration'].quantile(0.95), 'health_status'] = 'Critical'
    
    # Add timestamp if not present
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
    
    st.session_state.df = df
    st.session_state.historical_data = df.copy()
    st.session_state.data_loaded = True

df = st.session_state.df

# ================================
# SIDEBAR CONFIGURATION
# ================================
st.sidebar.header("⚙️ System Control Panel")

# Theme Toggle
st.sidebar.subheader("🎨 Theme Selection")
theme_mode = st.sidebar.radio(
    "Choose Theme",
    ["🌙 Dark Mode", "☀️ Light Mode"],
    index=0 if st.session_state.theme_mode == 'dark' else 1
)
st.session_state.theme_mode = 'dark' if theme_mode == "🌙 Dark Mode" else 'light'
colors = get_theme_colors()

# User Profile
st.sidebar.subheader("👤 User Profile")
st.sidebar.write(f"**Name:** {st.session_state.user_profile['name']}")
st.sidebar.write(f"**Role:** {st.session_state.user_profile['role']}")
st.sidebar.write(f"**Level:** {st.session_state.user_profile['level']}")

# Achievements Display
if st.session_state.achievements:
    st.sidebar.subheader("🏆 Achievements")
    for achievement in st.session_state.achievements:
        st.sidebar.markdown(f'<span class="achievement-badge">🎖️ {achievement}</span>', 
                          unsafe_allow_html=True)

# Operating Mode
st.sidebar.subheader("🎯 Operating Mode")
mode = st.sidebar.selectbox(
    "Select Dashboard Mode",
    [
        "📊 Dashboard Analysis",
        "⚡ Real-Time Monitoring",
        "🔮 Predictive Maintenance",
        "🏢 Fleet Management",
        "🌐 Digital Twin",
        "🎮 Gamification",
        "📱 Mobile Preview",
        "⚙️ System Settings"
    ],
    index=0
)

# Elevator Selection
st.sidebar.subheader("🏢 Elevator Selection")
elevator_list = sorted(df['elevator_id'].unique()) if 'elevator_id' in df.columns else ['All Elevators']
selected_elevator = st.sidebar.selectbox("Select Elevator", elevator_list)

# Filter data by selected elevator
if selected_elevator != 'All Elevators' and 'elevator_id' in df.columns:
    df_filtered = df[df['elevator_id'] == selected_elevator]
else:
    df_filtered = df

# Real-Time Controls
vibration_threshold = st.session_state.get('vibration_threshold', df_filtered['vibration'].quantile(0.90))
if mode == "⚡ Real-Time Monitoring":
    st.sidebar.subheader("🔄 Real-Time Controls")
    auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
    vibration_threshold = st.sidebar.slider("🚨 Vibration Alert Threshold", 
                                          float(df_filtered['vibration'].min()),
                                          float(df_filtered['vibration'].max()),
                                          float(vibration_threshold))
    st.session_state.vibration_threshold = vibration_threshold

# System Health Monitor
st.sidebar.subheader("💻 System Health")
st.sidebar.progress(st.session_state.system_health['cpu_usage'] / 100, "CPU")
st.sidebar.progress(st.session_state.system_health['memory_usage'] / 100, "Memory")
st.sidebar.progress(st.session_state.system_health['disk_usage'] / 100, "Disk")

# Language Selection
st.sidebar.subheader("🌍 Language")
language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French", "German", "Chinese"])

# ================================
# MAIN HEADER WITH IMAGE
# ================================

# Display banner image
try:
    st.image("generated_images/generated_image_d7f9a95d-0328-4dc3-b492-b24fc9f518f5_0.png", 
             use_column_width=True, caption="Smart Building IoT Network")
except:
    pass

st.markdown(f"""
<div class="main-header">
    <h1>🛗 ULTIMATE Smart Elevator Monitoring System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        🚀 AI-Powered Predictive Maintenance | ⚡ Real-Time Analytics | 🌐 Digital Twin Technology | 
        🎮 Gamified Experience | 🏢 Fleet Management
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
        Powered by Advanced Machine Learning • IoT Integration • Cloud Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Notification Badge
if st.session_state.notifications:
    latest_notification = st.session_state.notifications[-1]
    badge_class = f"alert-{latest_notification['level']}"
    st.markdown(f"""
    <div class="notification-badge {badge_class}">
        🔔 {latest_notification['message']} ({latest_notification['timestamp']})
    </div>
    """, unsafe_allow_html=True)

# Real-time Clock and Weather
col1, col2 = st.columns(2)

with col1:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"""
    <div class="clock-widget">
        🕐 {current_time}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="weather-widget">
        🌤️ New York: 72°F | Humidity: 65% | Wind: 8 mph
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ================================
# DASHBOARD ANALYSIS MODE
# ================================
if mode == "📊 Dashboard Analysis":
    st.header("📊 Comprehensive Dashboard Analysis")
    
    # KPI Metrics with animations
    st.subheader("🎯 Real-Time Performance Metrics")
    
    current_vibration = df_filtered['vibration'].iloc[-1] if len(df_filtered) > 0 else 0
    avg_vibration = df_filtered['vibration'].mean()
    health_score = calculate_health_score(current_vibration, 
                                         df_filtered['temperature'].iloc[-1] if 'temperature' in df_filtered.columns else 25,
                                         df_filtered['humidity'].iloc[-1])
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        health_color = "🟢" if health_score > 70 else "🟡" if health_score > 40 else "🔴"
        st.metric(f"{health_color} Health Score", f"{health_score:.1f}%", 
                 delta=f"{(current_vibration - avg_vibration):.3f}")
    
    with col2:
        st.metric("Vibration", f"{current_vibration:.3f} mm/s",
                 delta=f"{(current_vibration - avg_vibration):.3f}")
    
    with col3:
        st.metric("Revolutions", f"{df_filtered['revolutions'].mean():.1f}")
    
    with col4:
        temp = df_filtered['temperature'].mean() if 'temperature' in df_filtered.columns else 25
        st.metric("Temperature", f"{temp:.1f}°C")
    
    with col5:
        critical_count = len(df_filtered[df_filtered['health_status'] == 'Critical'])
        st.metric("Critical Alerts", critical_count)
    
    with col6:
        total_cycles = df_filtered['revolutions'].sum() if 'revolutions' in df_filtered.columns else 0
        st.metric("Total Cycles", f"{total_cycles:,.0f}")
    
    st.divider()
    
    # Advanced Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Multi-Sensor Time Series Analysis")
        
        fig1 = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Vibration Over Time', 'Humidity Trends', 
                           'Revolutions Count', 'Temperature Variations',
                           'Voltage Fluctuations', 'Energy Consumption'),
            vertical_spacing=0.08
        )
        
        sample_size = min(1000, len(df_filtered))
        
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['vibration'][:sample_size],
                       name='Vibration', line=dict(color='#ef4444')),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['humidity'][:sample_size],
                       name='Humidity', line=dict(color='#3b82f6')),
            row=1, col=2
        )
        
        fig1.add_trace(
            go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['revolutions'][:sample_size],
                       name='Revolutions', line=dict(color='#10b981')),
            row=2, col=1
        )
        
        if 'temperature' in df_filtered.columns:
            fig1.add_trace(
                go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['temperature'][:sample_size],
                           name='Temperature', line=dict(color='#f59e0b')),
                row=2, col=2
            )
        
        if 'voltage' in df_filtered.columns:
            fig1.add_trace(
                go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['voltage'][:sample_size],
                           name='Voltage', line=dict(color='#8b5cf6')),
                row=3, col=1
            )
        
        if 'energy_consumption' in df_filtered.columns:
            fig1.add_trace(
                go.Scatter(x=df_filtered.index[:sample_size], y=df_filtered['energy_consumption'][:sample_size],
                           name='Energy', line=dict(color='#ec4899')),
                row=3, col=2
            )
        
        fig1.update_layout(height=800, showlegend=False, 
                          title_text="Real-time Multi-Sensor Monitoring Dashboard")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
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
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
        
        fig2.update_layout(
            title="Feature Correlation Heatmap",
            width=500, height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Technical Diagram
    try:
        st.subheader("🔧 Sensor Network Architecture")
        st.image("generated_images/generated_image_fe8e8bb1-172e-4b01-8e4c-2dc529ea2bc3_0.png", 
                use_column_width=True, caption="Elevator Sensor Placement Diagram")
    except:
        pass
    
    # Statistical Analysis
    st.subheader("📊 Statistical Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vibration Distribution', 'Humidity Distribution',
                           'Revolutions Distribution', 'Health Status Distribution')
        )
        
        fig3.add_trace(go.Histogram(x=df_filtered['vibration'], name='Vibration',
                                    marker_color='#ef4444', nbinsx=50), row=1, col=1)
        fig3.add_trace(go.Histogram(x=df_filtered['humidity'], name='Humidity',
                                    marker_color='#3b82f6', nbinsx=50), row=1, col=2)
        fig3.add_trace(go.Histogram(x=df_filtered['revolutions'], name='Revolutions',
                                    marker_color='#10b981', nbinsx=50), row=2, col=1)
        
        health_counts = df_filtered['health_status'].value_counts()
        fig3.add_trace(go.Bar(x=health_counts.index, y=health_counts.values,
                             name='Health Status', marker_color=['#10b981', '#f59e0b', '#ef4444']),
                      row=2, col=2)
        
        fig3.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("🏅 Performance Analysis")
        
        # Create performance metrics
        performance_data = {
            'Metric': ['Vibration Stability', 'Temperature Control', 'Energy Efficiency', 
                      'Door Performance', 'Motor Efficiency', 'Overall Score'],
            'Score': [85, 92, 78, 88, 95, 87],
            'Trend': ['↑', '→', '↑', '→', '↑', '↑']
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        fig4 = go.Figure(data=[
            go.Bar(name='Score', x=perf_df['Metric'], y=perf_df['Score'],
                  marker_color=['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'])
        ])
        
        fig4.update_layout(
            title="Performance Metrics Overview",
            yaxis_title="Score (%)",
            height=500
        )
        st.plotly_chart(fig4, use_container_width=True)

# ================================
# REAL-TIME MONITORING MODE
# ================================
elif mode == "⚡ Real-Time Monitoring":
    st.header("⚡ Real-Time Monitoring Dashboard")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    
    # Live Sensor Gauges
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
                'bar': {'color': "#ef4444"},
                'steps': [
                    {'range': [0, vibration_threshold], 'color': "lightgray"},
                    {'range': [vibration_threshold, df_filtered['vibration'].max()], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 4},
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
                'bar': {'color': "#f59e0b"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 35], 'color': "gray"},
                    {'range': [35, 50], 'color': "#fee2e2"}
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
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "#dbeafe"}
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
                'bar': {'color': "#10b981"},
                'steps': [
                    {'range': [0, df_filtered['revolutions'].quantile(0.5)], 'color': "lightgray"},
                    {'range': [df_filtered['revolutions'].quantile(0.5), df_filtered['revolutions'].max()], 'color': "gray"}
                ]
            }
        ))
        fig_revs.update_layout(height=300)
        st.plotly_chart(fig_revs, use_container_width=True)
    
    # Real-time Alerts
    st.subheader("🚨 Real-Time Alert System")
    
    current_vibration = latest_data['vibration']
    alerts = []
    
    if current_vibration > vibration_threshold:
        alerts.append({
            'type': 'critical',
            'message': f'⚠️ CRITICAL: Vibration ({current_vibration:.3f}) exceeds threshold ({vibration_threshold:.3f})',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        add_notification(f'Critical vibration alert: {current_vibration:.3f}', 'critical')
    
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
        <div class="alert-success">
            ✅ All systems operating within normal parameters
        </div>
        """, unsafe_allow_html=True)

# ================================
# PREDICTIVE MAINTENANCE MODE
# ================================
elif mode == "🔮 Predictive Maintenance":
    st.header("🔮 Predictive Maintenance & AI Analytics")
    
    # Train ML Model
    if st.button("🤖 Train Advanced Predictive Models"):
        st.session_state.model_trained = True
        add_notification("ML models trained successfully!", 'success')
        
        with st.spinner("Training advanced machine learning models..."):
            # Prepare data
            features = ['vibration', 'humidity', 'revolutions', 'x1', 'x2', 'x3', 'x4', 'x5']
            if 'temperature' in df_filtered.columns:
                features.append('temperature')
            if 'voltage' in df_filtered.columns:
                features.append('voltage')
            
            X = df_filtered[features].dropna()
            y = (df_filtered.loc[X.index, 'vibration'] > df_filtered['vibration'].quantile(0.75)).astype(int)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Train Gradient Boosting
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb_model.fit(X, y)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X)
            
            # Store models
            st.session_state.rf_model = rf_model
            st.session_state.gb_model = gb_model
            st.session_state.iso_forest = iso_forest
            st.session_state.features = features
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.session_state.feature_importance = feature_importance
            
        st.success("✅ Advanced ML Models trained successfully!")
        add_achievement("ML Model Trainer")
    
    # Display results
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
        
        if 'voltage' in df_filtered.columns:
            components['Power System'] = abs(df_filtered['voltage'].mean() - 230) / 50
        
        for component, score in components.items():
            health_percentage = (1 - score) * 100
            health_color = "#10b981" if health_percentage > 70 else "#f59e0b" if health_percentage > 40 else "#ef4444"
            
            st.markdown(f"""
            <div class="stats-card" style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>📋 {component}</strong>
                    <span style="color: {health_color}; font-weight: bold; font-size: 1.2rem;">{health_percentage:.1f}%</span>
                </div>
                <div style="margin-top: 10px;">
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {health_percentage}%; background: {health_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================================
# FLEET MANAGEMENT MODE
# ================================
elif mode == "🏢 Fleet Management":
    st.header("🏢 Multi-Elevator Fleet Management")
    
    # Fleet Overview
    elevator_stats = []
    for elevator in sorted(df['elevator_id'].unique()) if 'elevator_id' in df.columns else ['All']:
        if elevator in df['elevator_id'].values:
            elevator_data = df[df['elevator_id'] == elevator]
            stats = {
                'Elevator': elevator,
                'Health Score': max(0, 100 - elevator_data['vibration'].mean() * 20),
                'Avg Vibration': elevator_data['vibration'].mean(),
                'Total Cycles': elevator_data['revolutions'].sum() if 'revolutions' in elevator_data.columns else 0,
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
                    format='%.1f%%',
                    min_value=0,
                    max_value=100
                ),
                'Critical Alerts': st.column_config.NumberColumn(
                    'Critical Alerts',
                    format='%d'
                )
            },
            use_container_width=True
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
    
    # Performance Comparison
    st.subheader("🔄 Elevator Performance Comparison")
    
    fig_comparison = go.Figure()
    
    for elevator in fleet_df['Elevator'][:5]:  # Show first 5
        elevator_data = fleet_df[fleet_df['Elevator'] == elevator]
        fig_comparison.add_trace(go.Scatter(
            x=['Vibration', 'Cycles (k)', 'Alerts'],
            y=[
                elevator_data['Avg Vibration'].values[0],
                elevator_data['Total Cycles'].values[0] / 1000,
                elevator_data['Critical Alerts'].values[0]
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

# ================================
# DIGITAL TWIN MODE
# ================================
elif mode == "🌐 Digital Twin":
    st.header("🌐 Digital Twin Visualization")
    
    st.subheader("🏢 3D Building Elevator Simulation")
    
    # Create 3D visualization
    fig_3d = go.Figure()
    
    # Simulate building floors
    floors = 20
    for floor in range(floors):
        # Floor
        fig_3d.add_trace(go.Mesh3d(
            x=[0, 10, 10, 0],
            y=[0, 0, 10, 10],
            z=[floor, floor, floor, floor],
            color='lightblue',
            opacity=0.1,
            showlegend=False
        ))
    
    # Elevator shafts
    for i, elevator in enumerate(elevator_list[:3]):
        current_floor = random.randint(0, floors - 1)
        
        # Elevator car
        fig_3d.add_trace(go.Mesh3d(
            x=[2 + i*4, 3 + i*4, 3 + i*4, 2 + i*4],
            y=[4, 4, 6, 6],
            z=[current_floor, current_floor, current_floor + 1, current_floor + 1],
            color=['#ef4444' if df[df['elevator_id'] == elevator]['health_status'].mode()[0] == 'Critical' else
                  '#10b981'],
            opacity=0.8,
            name=elevator,
            showlegend=True
        ))
    
    fig_3d.update_layout(
        title="Real-Time Digital Twin - Elevator Positions",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Floor Level',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Digital Twin Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Elevators", len(elevator_list))
    
    with col2:
        st.metric("Total Floors", floors)
    
    with col3:
        st.metric("Simulated Time", datetime.now().strftime('%H:%M:%S'))
    
    with col4:
        st.metric("Sync Status", "✅ Online")

# ================================
# GAMIFICATION MODE
# ================================
elif mode == "🎮 Gamification":
    st.header("🎮 Gamification & Achievements")
    
    # User Profile Card
    st.subheader("👤 User Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h2>{st.session_state.user_profile['name']}</h2>
            <p><strong>Role:</strong> {st.session_state.user_profile['role']}</p>
            <p><strong>Level:</strong> {st.session_state.user_profile['level']}</p>
            <p><strong>Experience:</strong> {st.session_state.user_profile['experience']} XP</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Progress to Next Level")
        xp_for_next_level = (st.session_state.user_profile['level'] + 1) * 500
        current_level_xp = st.session_state.user_profile['level'] * 500
        xp_progress = (st.session_state.user_profile['experience'] - current_level_xp) / (xp_for_next_level - current_level_xp) * 100
        
        st.markdown(f"""
        <div class="progress-container" style="height: 30px;">
            <div class="progress-bar" style="width: {xp_progress}%;"></div>
        </div>
        <p style="text-align: center; margin-top: 10px;">
            {xp_progress:.1f}% to Level {st.session_state.user_profile['level'] + 1}
        </p>
        """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("🏆 Leaderboard")
        leaderboard_data = pd.DataFrame({
            'Rank': [1, 2, 3, 4, 5],
            'User': ['Admin User', 'Engineer A', 'Tech Lead', 'Manager B', 'Operator C'],
            'XP': [1500, 1420, 1350, 1280, 1200],
            'Level': [5, 4, 4, 4, 3]
        })
        st.dataframe(leaderboard_data, use_container_width=True)
    
    # Achievements Section
    st.subheader("🏅 Achievements")
    
    all_achievements = [
        ("First Step", "Login for the first time", True),
        ("Data Master", "Upload your first dataset", True),
        ("Analyst", "Create your first analysis", True),
        ("Predictor", "Train your first ML model", False),
        ("Fleet Manager", "Manage 5+ elevators", False),
        ("Dedication", "Use the app for 7 days", False),
        ("Expert", "Reach Level 10", False),
        ("Master", "Reach Level 20", False),
        ("Guardian", "Prevent 10 failures", False),
        ("Optimizer", "Reduce energy usage by 10%", False)
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for achievement, desc, unlocked in all_achievements[:5]:
            status = "✅" if unlocked else "🔒"
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                        background: {colors['card_bg']}; border: 2px solid {colors['border']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{status} {achievement}</strong>
                </div>
                <p style="font-size: 0.9rem; margin-top: 0.5rem; color: {colors['text']}; opacity: 0.7;">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for achievement, desc, unlocked in all_achievements[5:]:
            status = "✅" if unlocked else "🔒"
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                        background: {colors['card_bg']}; border: 2px solid {colors['border']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{status} {achievement}</strong>
                </div>
                <p style="font-size: 0.9rem; margin-top: 0.5rem; color: {colors['text']}; opacity: 0.7;">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Daily Challenges
    st.subheader("🎯 Daily Challenges")
    
    challenges = [
        ("Analyze 3 different elevators", "📊", 50),
        ("Train ML model", "🤖", 100),
        ("Export report", "📤", 25),
        ("Monitor for 10 minutes", "⏱️", 75)
    ]
    
    for challenge, icon, xp in challenges:
        st.markdown(f"""
        <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                    background: linear-gradient(135deg, {colors['accent']}22, {colors['accent']}44); 
                    border: 2px solid {colors['accent']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>{icon} {challenge}</span>
                <span style="color: {colors['accent']}; font-weight: bold;">+{xp} XP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================
# MOBILE PREVIEW MODE
# ================================
elif mode == "📱 Mobile Preview":
    st.header("📱 Mobile Preview Mode")
    
    st.info("This mode shows how the dashboard would look on a mobile device.")
    
    # Simulate mobile view
    st.markdown(f"""
    <div style="max-width: 375px; margin: 0 auto; background: {colors['card_bg']}; 
                border-radius: 30px; padding: 20px; border: 3px solid {colors['border']};
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
        <h3 style="text-align: center; margin-bottom: 20px;">🛗 Mobile Dashboard</h3>
        
        <div class="metric-card" style="margin-bottom: 15px;">
            <h4>Health Score</h4>
            <h2 style="color: {colors['success']};">85.3%</h2>
        </div>
        
        <div class="metric-card" style="margin-bottom: 15px;">
            <h4>Vibration</h4>
            <h2 style="color: {colors['warning']};">0.52 mm/s</h2>
        </div>
        
        <div class="metric-card" style="margin-bottom: 15px;">
            <h4>Status</h4>
            <h2 style="color: {colors['success']};">✅ Normal</h2>
        </div>
        
        <div class="alert-info" style="margin-top: 20px;">
            <strong>ℹ️ Quick Actions:</strong><br>
            • View Analytics<br>
            • Schedule Maintenance<br>
            • Export Report
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# SYSTEM SETTINGS MODE
# ================================
elif mode == "⚙️ System Settings":
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔔 Notification Settings")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
        push_notifications = st.checkbox("Push Notifications", value=True)
        
        st.subheader("🎨 Display Settings")
        
        animation_speed = st.slider("Animation Speed", 1, 10, 5)
        chart_theme = st.selectbox("Chart Theme", ["Plotly", "Seaborn", "Default"])
        
    with col2:
        st.subheader("🔒 Security Settings")
        
        two_factor_auth = st.checkbox("Two-Factor Authentication")
        session_timeout = st.slider("Session Timeout (minutes)", 15, 120, 30)
        
        st.subheader("📊 Data Settings")
        
        auto_backup = st.checkbox("Automatic Backup", value=True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        
    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")
        add_notification("Settings updated", 'success')

# ================================
# EXPORT FUNCTIONALITY
# ================================
st.divider()
st.header("📤 Export & Reporting")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Export Analysis Report"):
        report_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Elevator': selected_elevator,
            'Total Samples': len(df_filtered),
            'Avg Vibration': df_filtered['vibration'].mean(),
            'Health Score': calculate_health_score(
                df_filtered['vibration'].iloc[-1] if len(df_filtered) > 0 else 0,
                df_filtered['temperature'].iloc[-1] if 'temperature' in df_filtered.columns else 25,
                df_filtered['humidity'].iloc[-1]
            ),
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
            label="Download Dataset (CSV)",
            data=csv,
            file_name=f"elevator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📋 Export JSON"):
        json_data = df_filtered.head(1000).to_json(orient='records')
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"elevator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col4:
    if st.button("📧 Email Report"):
        st.info("📧 Report sent to admin@company.com")
        add_notification("Email report sent successfully", 'success')

# ================================
# FOOTER WITH LOGO
# ================================
st.divider()

# Try to load and display logo
try:
    with open("generated_images/generated_image_b6d72252-d0e5-46e5-8c24-c96e755edc29_0.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    st.markdown(f"""
    <div class="logo-container">
        <h3 style="margin-bottom: 1rem;">Smart Elevator Monitoring System</h3>
        <img src="data:image/png;base64,{encoded_image}" 
             alt="Elevator Logo" style="max-width: 150px;">
        <p style="margin-top: 1rem; color: {colors['text']}; opacity: 0.7;">
            🛗 Ultimate Smart Building Solutions • AI-Powered Predictive Maintenance • Real-Time Analytics
        </p>
        <p style="font-size: 0.8rem; color: {colors['text']}; opacity: 0.5;">
            © 2024 Smart Building Technologies | Version 3.0 Ultimate Edition
        </p>
    </div>
    """, unsafe_allow_html=True)
except:
    st.markdown(f"""
    <div class="logo-container">
        <h3 style="margin-bottom: 1rem;">Smart Elevator Monitoring System</h3>
        <p style="margin-top: 1rem; color: {colors['text']}; opacity: 0.7;">
            🛗 Ultimate Smart Building Solutions • AI-Powered Predictive Maintenance • Real-Time Analytics
        </p>
        <p style="font-size: 0.8rem; color: {colors['text']}; opacity: 0.5;">
            © 2024 Smart Building Technologies | Version 3.0 Ultimate Edition
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# KEYBOARD SHORTCUTS INFO
# ================================
with st.expander("⌨️ Keyboard Shortcuts"):
    st.markdown("""
    - **Ctrl + B**: Open/Close Sidebar
    - **Ctrl + D**: Toggle Dark/Light Mode
    - **Ctrl + R**: Refresh Data
    - **Ctrl + E**: Export Current View
    - **Ctrl + H**: Show Help
    - **Ctrl + N**: New Notification
    - **Ctrl + S**: Save Settings
    """)

# ================================
# END OF APPLICATION
# ================================
