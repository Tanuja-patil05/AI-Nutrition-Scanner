# AI Nutrition Scanner 🥗

An intelligent food identification and nutritional analysis dashboard built with Streamlit and TensorFlow.

## Screenshots

![App Dashboard Mockup](https://raw.githubusercontent.com/streamlit/template-nutrition-tracker/main/screenshot.png)

## Features
- **Deep Learning Vision**: Uses EfficientNetB0 Transfer Learning to identify 101 food classes.
- **Instant Macro Retrieval**: Integrated with CalorieNinjas API for real-time calorie, protein, fat, and carb data.
- **Health Scoring**: Proprietary algorithm to evaluate meal quality.
- **Interactive Visuals**: Plotly-powered macro breakdowns and daily tracking logs.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- pip (package manager)

### 2. Installation
```powershell
pip install -r requirements.txt
```

### 3. Training & Weights
For best accuracy, you should provide a trained model:
1. Place your `model.h5` in the project root.
2. The app will automatically attempt to load it.

### 4. API Key (Optional)
This app works with mock data by default. To use real-time data:
1. Get a free API Key from [CalorieNinjas](https://calorieninjas.com/api).
2. Set it as an environment variable:
   ```powershell
   $env:CALORIE_NINJAS_API_KEY = "your_key_here"
   ```

### 5. Running the App
```powershell
streamlit run app.py
```

## Architecture
- **Frontend**: Streamlit
- **ML Engine**: TensorFlow (EfficientNetB0 Transfer Learning)
- **Image Processing**: OpenCV & PIL
- **Data Visuals**: Plotly
- **API**: CalorieNinjas

## Disclaimer
This application is for educational purposes. Nutritional values are estimates and should not be used for medical advice.
