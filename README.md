🛗 Smart Elevator Monitoring Dashboard

A comprehensive predictive maintenance dashboard for elevator systems using machine learning and real-time sensor data analysis.

📋 Table of Contents

- 🎯 Overview

- ✨ Features

- 🚀 Installation

- 🎯 Usage

- 📊 Dataset Requirements

- 🔧 Technical Details

- 🎨 Customization

- 🐛 Troubleshooting

- 🚀 Deployment

- 📄 License

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

🚀 Installation

Prerequisites

- Python 3.8 or higher

- pip package manager

Step 1: Clone or Download the Project

git clone <repository-url>
cd smart-elevator-monitoring-dashboard

Step 2: Create Virtual Environment (Optional but Recommended)

python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Step 3: Install Required Packages

Create a requirements.txt file with the following content:

streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0

Install the packages:

pip install -r requirements.txt

Step 4: Prepare Dataset

Ensure your dataset file Elevator predictive-maintenance-dataset.csv is in the same directory as app.py.

The dataset must contain the following columns:

- ID

- revolutions

- humidity

- vibration

- x1, x2, x3, x4, x5 (sensor readings)

Step 5: Add Logo (Optional)

Place your logo file LIFTBOT.png in the project directory. If you don't have a logo, the app will still work without it.

🎯 Usage

Running the Dashboard

Start the Streamlit application:

streamlit run app.py

The dashboard will automatically open in your default web browser at http://localhost:8501

Navigation

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

📊 Dataset Requirements

Required Columns

 Column Description Type Example

 ID Unique identifier Integer/String 1, 2, 3...

 revolutions Motor revolutions Numeric 1500, 1600, 1550

 humidity Humidity level (%) Numeric 45.2, 46.1, 44.8

 vibration Vibration measurement Numeric 0.123, 0.145, 0.118

 x1-x5 Additional sensor readings Numeric 0.456, 0.789, 0.234

Data Quality

- The dashboard automatically handles:

- Duplicate rows (removed)

- Missing values (rows removed)

- Non-numeric values (converted or removed)

- Recommended minimum: 100+ rows for meaningful analysis

- Recommended maximum: 100,000 rows for optimal performance

Example Dataset Structure

ID,revolutions,humidity,vibration,x1,x2,x3,x4,x5
1,1500,45.2,0.123,0.456,0.789,0.234,0.567,0.890
2,1600,46.1,0.145,0.467,0.790,0.245,0.578,0.901
3,1550,44.8,0.118,0.445,0.778,0.222,0.555,0.888
4,1520,45.5,0.130,0.450,0.780,0.230,0.560,0.895
5,1580,45.8,0.138,0.460,0.785,0.235,0.565,0.898
...

🔧 Technical Details

Machine Learning Models

Random Forest Classifier

- Purpose: Predict high vibration events

- Threshold: 75th percentile of vibration values

- Features: revolutions, humidity, x1, x2, x3, x4, x5

- Parameters:

- n_estimators: 100 (number of trees)

- random_state: 42 (reproducibility)

- Output: Binary classification (normal/high vibration)

Isolation Forest

- Purpose: Detect anomalous patterns

- Contamination: 0.1 (10% expected anomaly rate)

- Features: Same as Random Forest

- Parameters:

- contamination: 0.1

- random_state: 42

- Output: Anomaly score and binary classification (-1 for anomaly, 1 for normal)

Performance Considerations

- Training Time: Typically 1-5 seconds for datasets up to 10,000 rows

- Memory Usage: Depends on dataset size

- Recommended Dataset Size: 1,000-100,000 rows

- Visualization Rendering: Instant for most charts

Technology Stack

- Frontend: Streamlit

- Data Processing: Pandas, NumPy

- Visualization: Plotly, Matplotlib, Seaborn

- Machine Learning: scikit-learn

- Styling: Custom CSS

🎨 Customization

Changing the Color Theme

The dashboard uses a blue color scheme. To customize:

- Open app.py

- Locate the CSS section in the st.markdown block

- Modify the color variables:

:root {
    --primary-color: #2563eb;    /* Main blue */
    --secondary-color: #1e40af;  /* Darker blue */
    --accent-color: #3b82f6;     /* Lighter blue */
    --light-blue: #dbeafe;       /* Background */
    --dark-blue: #1e3a8a;        /* Text */
}

Adjusting ML Model Parameters

Modify model parameters in the Predictive Maintenance section:

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees (try 50-200)
    random_state=42
)

# Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,     # Expected anomaly rate (0.05-0.2)
    random_state=42
)

Changing Vibration Threshold Percentile

Default is 75th percentile. To change:

threshold_75th = df['vibration'].quantile(0.75)  # Change 0.75 to desired value
# Examples: 0.5 (median), 0.9 (90th percentile)

Customizing Chart Colors

Modify color sequences in Plotly charts:

# Line plot
color_discrete_sequence=['#2563eb']  # Change to your color

# Histogram
color_discrete_sequence=['#3b82f6']  # Change to your color

# Box plot
blue_colors = ['#1e3a8a', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa']
# Modify these hex codes

🐛 Troubleshooting

Common Issues

Issue: "Dataset Not Found" error

Solution: Ensure Elevator predictive-maintenance-dataset.csv is in the same directory as app.py

Issue: "Missing Required Columns" error

Solution: Verify your CSV contains all required columns: ID, revolutions, humidity, vibration, x1, x2, x3, x4, x5

Issue: Logo not displaying

Solution: Ensure LIFTBOT.png exists in the project directory, or remove the logo line from the code

Issue: Models not training

Solution: Ensure you have sufficient data (minimum 50 rows) and all numeric columns contain valid numbers

Issue: Charts not displaying

Solution: Check browser console for errors, ensure all dependencies are installed correctly

Issue: Slow performance

Solution: Reduce dataset size or increase system resources. For large datasets (>100,000 rows), consider data sampling

Issue: Port already in use

Solution: Specify a different port when running:

streamlit run app.py --server.port 8501

Debug Mode

Enable debug mode for detailed error messages:

streamlit run app.py --logger.level=debug

🚀 Deployment

Local Deployment

streamlit run app.py

Cloud Deployment (Streamlit Cloud)

- Push code to GitHub repository

- Connect repository to Streamlit Cloud

- Deploy automatically

Docker Deployment

Create a Dockerfile:

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

Build and run:

docker build -t elevator-dashboard .
docker run -p 8501:8501 elevator-dashboard

Heroku Deployment

- Create Procfile:

web: streamlit run app.py --server.port=$PORT

- Create requirements.txt

- Deploy:

heroku create your-app-name
git push heroku main

AWS Deployment

- Launch EC2 instance

- Install dependencies

- Clone repository

- Run with systemd service

Environment Variables

Set environment variables for configuration:

export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

📄 License

This project is licensed under the MIT License - see LICENSE file for details.

🙏 Acknowledgments

- Built with Streamlit

- Machine learning powered by scikit-learn

- Visualizations created with Plotly

- Data processing with Pandas

📧 Support

For issues, questions, or contributions:

- Open an issue in the project repository

- Contact the development team

- Refer to Streamlit documentation: https://docs.streamlit.io

Version: 1.0.0
Last Updated: 2024
Maintained by: Smart Elevator Monitoring Team

🛗 Elevating Maintenance Through Intelligence

⬆ Back to Top
