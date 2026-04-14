# Explainable Finance AI - Setup & Run Instructions

## Project Overview
This is a Flask-based web application for credit risk prediction using Machine Learning models (Random Forest and Gradient Boosting). It provides explainable AI predictions with SHAP visualizations.

---

## Prerequisites
- Python 3.11+ installed on the system
- Virtual environment (venv) already configured in the project folder

---

## Setup Instructions (Run Once)

### 1. Activate Virtual Environment
```powershell
# Windows PowerShell
& ./venv/Scripts/Activate.ps1

# Or use Command Prompt
venv\Scripts\activate.bat
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Train/Retrain Models (if needed)
If model pickle files are missing or corrupted:
```powershell
python train_models.py
```

**Important**: Always run `train_models.py` first if you're using a new version of scikit-learn to avoid `ModuleNotFoundError` during model loading.

---

## Running the Project

### Start the Flask Application
```powershell
python app.py
```

### Expected Output
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Running on http://10.61.138.147:5000
Press CTRL+C to quit
```

### Access the Application
Open your browser and navigate to:
- **Local**: http://127.0.0.1:5000
- **Network**: http://10.61.138.147:5000

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sklearn.ensemble._gb_losses'`
**Solution**: Retrain the models to match your scikit-learn version:
```powershell
python train_models.py
```

### Issue: Port 5000 already in use
**Solution**: Modify port in app.py (line ~100):
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001 or another port
```

### Issue: Database connection errors
**Solution**: The app automatically creates SQLite database on first run. Ensure write permissions in project folder.

---

## Project Structure
```
explainable-finance-ai/
├── app.py                          # Main Flask application entry point
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── app/
│   ├── services/predictor.py      # ML prediction logic
│   ├── database/db.py             # Database operations
│   ├── data/german_credit_data.csv # Training dataset
│   ├── models/                     # Trained model files (.pkl)
│   └── templates/                  # HTML templates
└── venv/                           # Virtual environment
```

---

## Model Information

### Random Forest Model
- **Accuracy**: ~95%
- **Precision**: ~93%
- **Recall**: ~90%
- **F1-Score**: ~92%

### Gradient Boosting Model
- **Accuracy**: ~84%
- **Precision**: ~79%
- **Recall**: ~62%
- **F1-Score**: ~69%

---

## Features
✅ Credit risk prediction (Good/Bad)
✅ Real-time model explanations (SHAP)
✅ Confusion matrices & ROC curves
✅ Model comparison visualizations
✅ Persistent result storage in SQLite
✅ RESTful API endpoints

---

## API Endpoints
- **GET** `/` - Main web interface
- **POST** `/api/predict/credit` - Credit prediction with JSON payload
- **GET** `/health` - Health check endpoint

---

## Quick Start Command
```powershell
# One-line to activate and run (after first setup)
& ./venv/Scripts/Activate.ps1; python app.py
```
