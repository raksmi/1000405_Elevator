🛗 Smart Elevator Monitoring Dashboard

A comprehensive predictive maintenance dashboard for elevator systems using machine learning and real-time sensor data analysis.

APP LINK: https://elevatormonitoringsystem.streamlit.app/

Also view https://www.canva.com/design/DAHBvM4ZVOQ/B-sqr3PHIeUhfkx6Hf1_NQ/edit?utm_content=DAHBvM4ZVOQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton for the storyboard

📋 Table of Contents

- 🎯 Overview

- ✨ Features

- 🎯 Usage

- 📊 Dataset Requirements

- 🔧 Technical Details

- 🎨 Credits

  
🎯 Overview

Smart Elevator Monitoring Dashboard is a sophisticated predictive maintenance application designed to monitor elevator systems through sensor data analysis. Built with Streamlit, it provides real-time visualization of elevator performance metrics including humidity, revolutions, and vibration, with machine learning-powered anomaly detection and predictive maintenance capabilities.

Key Highlights:

- 🎨 Modern Blue Theme: Professional, easy-to-read interface with consistent blue color scheme

- 📊 Real-Time Analytics: Interactive visualizations for comprehensive data analysis

- 🤖 Machine Learning Integration: Random Forest and Isolation Forest models for predictive maintenance

- 📈 Advanced Visualizations: 6 different chart types including line plots, histograms, scatter plots, box plots, and correlation heatmaps

- ⚙️ Interactive Thresholds: Adjustable vibration threshold slider for custom alert levels

- 🎯 Dual Mode Operation: Dashboard Analysis and Predictive Maintenance modes

- 📱 Responsive Design: Beautiful, mobile-friendly interface

- 🔍 Anomaly Detection: Automatic identification of unusual sensor patterns

✨ Features

📊 Dashboard Analysis Mode

📌 Key Performance Metrics

- Average Vibration: Mean vibration level across all readings

- Maximum Vibration: Highest recorded vibration value

- Average Humidity: Mean humidity percentage

- Total Revolutions: Cumulative motor revolutions

⚙️ Vibration Threshold Filter

- Interactive Slider: Adjust vibration alert threshold in real-time

- Dynamic Filtering: Automatically identify high-vibration samples

- Alert System: Visual warnings when samples exceed threshold

- Contextual Information: Count and percentage of samples above threshold

📈 Exploratory Data Analysis Visualizations

- Vibration Over Time (Line Plot)

- Shows vibration trends across all readings

- Red dashed line indicates custom threshold

- Helps identify periods of elevated vibration

- Humidity Distribution (Histogram)

- 50-bin histogram for detailed distribution

- Understand environmental conditions

- Identify normal operating ranges

- Revolutions Distribution (Histogram)

- Analyze motor usage patterns

- Identify frequency of revolution counts

- Spot outliers in motor performance

- Revolutions vs Vibration (Scatter Plot)

- Correlation analysis with OLS trendline

- Visual relationship between motor usage and vibration

- Threshold line for easy reference

- Sensor Variability & Outliers (Box Plot)

- Compare readings across 5 sensors (x1-x5)

- Identify outliers and inconsistent sensor behavior

- Assess sensor reliability and calibration

- Color-coded by sensor for easy identification

- Correlation Heatmap

- Visual representation of relationships between all numeric features

- Values range from -1 (negative correlation) to +1 (positive correlation)

- Helps identify which factors are most related

- Blue color scale for consistent theming

🔮 Predictive Maintenance Mode

🤖 Machine Learning Models

- Random Forest Classifier: Predicts high vibration events

- Uses 75th percentile as threshold

- Features: revolutions, humidity, x1, x2, x3, x4, x5

- 100 estimators for robust predictions

- Isolation Forest: Detects anomalous patterns

- 10% contamination rate

- Identifies unusual sensor readings

- Provides anomaly scores

📊 Feature Importance Analysis

- Horizontal Bar Chart: Visual ranking of feature importance

- Color Gradient: Blues scale for easy interpretation

- Identifies Key Factors: Shows which sensors most influence vibration

🎯 Anomaly Detection Results

- Scatter Plot: Visual representation of anomaly scores

- Color Coding: Red for anomalies, blue for normal

- Anomaly Count: Total number of detected anomalies

- Detailed View: Table of all anomalous samples

📋 Anomalous Samples

- Interactive Table: View all detected anomalies

- Complete Data: All sensor readings for anomalous samples

- Easy Export: Copy or download for further analysis

🎨 Interface Features

🌈 Blue Theme Design

- Primary Color: #2563eb (Bright Blue)

- Secondary Color: #1e40af (Dark Blue)

- Accent Color: #3b82f6 (Medium Blue)

- Light Blue: #dbeafe (Background)

- Dark Blue: #1e3a8a (Text)

✨ Visual Effects

- Gradient Headers: Beautiful blue gradient sections

- Card Styling: White cards with blue borders

- Alert Boxes: Color-coded information, warning, and success messages

- Hover Effects: Interactive feedback on buttons and elements

- Smooth Transitions: Animated state changes

📱 Responsive Design

- Wide Layout: Optimized for desktop viewing

- Flexible Columns: Adaptive 2, 3, and 4-column layouts

- Touch-Friendly: Optimized buttons and sliders

- Readable Fonts: Clear, high-contrast text

🔒 Data Quality Features

📋 Data Validation

- Required Columns Check: Ensures all necessary columns exist

- Missing Column Alerts: Clear error messages for missing data

- Automatic Validation: Stops execution if data is invalid

🧹 Data Cleaning

- Duplicate Removal: Automatically removes duplicate rows

- Missing Value Handling: Removes rows with null values

- Type Conversion: Converts columns to numeric types

- Quality Reporting: Shows number of duplicates and missing values removed

📊 Dataset Summary

- Dataset Info: Row count, column count, duplicates removed

- Column Names: List of all available columns

- Data Statistics: Missing values, numeric columns, data quality status


Dashboard Analysis Mode (Default)

- View Key Metrics: Check the 4 performance cards at the top

- Adjust Threshold: Use the slider to set vibration alert threshold

- Review Alerts: Check for high-vibration warnings

- Explore Visualizations: Scroll through all 6 charts

- Line plot for vibration trends

- Histograms for humidity and revolutions

- Scatter plot for correlations

- Box plot for sensor variability

- Heatmap for feature correlations

Predictive Maintenance Mode

- Train Models: Click "🤖 Train Predictive Models" button

- Wait for Training: Models train in 1-5 seconds (depending on data size)

- View Results:

- Feature importance bar chart

- Anomaly detection scatter plot

- Anomaly count and statistics

- Examine Anomalies: Review the table of anomalous samples

Understanding the Visualizations

Line Plot - Vibration Over Time

- Shows vibration trends across all readings

- Red dashed line indicates your custom threshold

- Helps identify periods of elevated vibration

- Hover over points for exact values

Histograms - Distribution Analysis

- Humidity: Understand environmental conditions

- Revolutions: Analyze motor usage patterns

- Identify normal operating ranges and outliers

- 50 bins for detailed distribution view

Scatter Plot - Revolutions vs Vibration

- Correlation analysis between motor usage and vibration

- Trendline shows overall relationship

- Threshold line highlights concerning readings

- Color-coded points for easy identification

Box Plot - Sensor Variability

- Compare readings across all 5 sensors (x1-x5)

- Identify outliers and inconsistent sensor behavior

- Assess sensor reliability and calibration

- Color-coded by sensor for easy comparison

Correlation Heatmap

- Visual representation of relationships between all numeric features

- Values range from -1 (negative correlation) to +1 (positive correlation)

- Helps identify which factors are most related

- Blue color scale for consistent theming



Technology Stack

- Frontend: Streamlit

- Data Processing: Pandas, NumPy

- Visualization: Plotly, Matplotlib, Seaborn

- Machine Learning: scikit-learn

- Styling: Custom CSS

Credits
Created by: M.Raksmi Priyasree (ID: 1000405)

Class: Artificial Intelligence: Mathematics in AI-I – Year 1

Mentor: Syed Ali Beema.S

School: Jain Vidyalaya IB world school, Madurai

