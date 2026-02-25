🛗 Smart Elevator Monitoring Dashboard

A comprehensive predictive maintenance dashboard for monitoring elevator sensor data with real-time analytics, anomaly detection, and machine learning-powered insights.

📋 Table of Contents

- Overview

- Features

- Usage

- Dashboard Modes

- Data Requirements

- Technical Stack

- File Structure



🎯 Overview

The Smart Elevator Monitoring Dashboard is a powerful web-based application designed to analyze elevator sensor readings including humidity, revolutions, and vibration data. It provides predictive maintenance capabilities through machine learning models, helping facility managers identify potential issues before they become critical problems.

Key Benefits

- Real-time Monitoring: Track elevator performance metrics in real-time

- Predictive Maintenance: Use ML models to predict potential failures

- Anomaly Detection: Identify unusual patterns in sensor data

- Interactive Visualizations: Explore data through interactive charts and graphs

- User-friendly Interface: Clean, intuitive blue-themed design

✨ Features

Core Features

- 📊 Dashboard Analysis Mode

- Key performance metrics (Average Vibration, Maximum Vibration, Average Humidity, Total Revolutions)

- Interactive vibration threshold filter

- Six comprehensive visualizations:

- Vibration trend over time (Line Plot)

- Humidity distribution (Histogram)

- Revolutions distribution (Histogram)

- Revolutions vs Vibration relationship (Scatter Plot)

- Sensor variability analysis (Box Plot)

- Correlation heatmap of all features

- 🔮 Predictive Maintenance Mode

- Machine Learning model training (Random Forest Classifier)

- Feature importance analysis

- Anomaly detection using Isolation Forest

- Real-time anomaly scoring

- Detailed anomalous sample identification

Design Features

- 🎨 Full Blue Theme: Professional blue color scheme throughout

- 📱 Responsive Design: Works on desktop and tablet devices

- 🖼️ Custom Logo: Professional elevator branding

- 👁️ High Visibility: All text clearly visible with proper contrast

- ⚡ Fast Performance: Optimized for quick data processing



📖 Usage


Using the Dashboard

- Select Mode: Choose between "Dashboard Analysis" or "Predictive Maintenance" from the sidebar

- Adjust Thresholds: Use the vibration threshold slider to filter high-vibration samples

- Train Models: In Predictive Maintenance mode, click "Train Predictive Models" to enable ML features

- Explore Visualizations: Interact with charts by hovering, zooming, and filtering

🎛️ Dashboard Modes

Mode 1: Dashboard Analysis

This mode provides comprehensive exploratory data analysis:

Key Metrics Section

- Average Vibration: Mean vibration level across all samples

- Maximum Vibration: Highest recorded vibration value

- Average Humidity: Mean humidity percentage

- Total Revolutions: Sum of all revolution counts

Vibration Threshold Filter

- Interactive slider to set alert thresholds

- Real-time filtering of high-vibration samples

- Visual alerts for samples exceeding threshold

Visualizations

- Vibration Over Time (Line Plot)

- Shows vibration trends across sample IDs

- Includes threshold reference line

- Identifies patterns and spikes

- Humidity Distribution (Histogram)

- Displays humidity value distribution

- 50-bin histogram for detailed analysis

- Blue-themed visualization

- Revolutions Distribution (Histogram)

- Shows revolution count distribution

- Identifies common operating ranges

- Helps detect outliers

- Revolutions vs Vibration (Scatter Plot)

- Correlation analysis between revolutions and vibration

- Includes trend line (OLS regression)

- Threshold line for reference

- Sensor Variability (Box Plot)

- Analyzes x1-x5 sensor readings

- Identifies outliers and variability

- Color-coded by sensor

- Correlation Heatmap

- Shows relationships between all numeric features

- Blue color scale for easy interpretation

- Correlation values displayed

Mode 2: Predictive Maintenance

This mode provides ML-powered insights:

Model Training

Click "Train Predictive Models" to:

- Train Random Forest Classifier for high vibration prediction

- Train Isolation Forest for anomaly detection

- Calculate feature importance scores

Feature Importance Analysis

- Horizontal bar chart showing most influential features

- Blue gradient color scheme

- Helps identify key factors affecting vibration

Anomaly Detection Results

- Scatter plot showing anomaly scores

- Color-coded: Blue (normal), Red (anomalous)

- Real-time anomaly count display

Anomalous Samples

- Detailed table of all detected anomalies

- Full sensor readings for each anomaly

- Exportable data for further analysis


🛠️ Technical Stack

Core Technologies

- Streamlit: Web application framework

- Python 3.11+: Programming language

- Pandas: Data manipulation and analysis

- NumPy: Numerical computing

Visualization Libraries

- Plotly: Interactive visualizations

- Plotly Graph Objects: Advanced chart customization

- Matplotlib: Backend plotting

- Seaborn: Statistical visualization

Machine Learning

- Scikit-learn: Machine learning library

- RandomForestClassifier: Classification model

- IsolationForest: Anomaly detection

- StandardScaler: Feature scaling

Styling

- CSS: Custom styling for blue theme

- HTML: Custom components and layouts

📁 File Structure

elevator-monitoring-dashboard/
│
├── elevator_dashboard.py      # Main Streamlit application
├── elevator_data.csv          # Sample sensor data
├── elevator_logo.png          # Application logo
├── README.md                  # This file
├── requirements.txt           # Python dependencies (optional)
│
└── outputs/                   # Generated outputs (auto-created)
    └── workspace_output_*.txt # Streamlit logs


