# QUICK START - EXAM REFERENCE CARD

## 30-Second Setup
```powershell
# Step 1: Activate virtual environment
& ./venv/Scripts/Activate.ps1

# Step 2: Train models (DO THIS FIRST - takes ~30 seconds)
python train_models.py

# Step 3: Run the application
python app.py
```

## Expected Result
✅ Terminal shows: `Running on http://127.0.0.1:5000`
✅ Open browser to: http://127.0.0.1:5000
✅ You should see the credit prediction interface

---

## What to Demonstrate

### 1. Web Interface Features
- Input credit parameters (Age, Job, Loan Amount, etc.)
- Click "Predict" button
- See predictions from 2 models: Random Forest & Gradient Boosting
- View model performance metrics (Accuracy, Precision, Recall, F1-Score)

### 2. Model Comparison
- Random Forest: ~95% accuracy (better performance)
- Gradient Boosting: ~84% accuracy (more conservative)

### 3. Explainability Features
- View confusion matrices visualization
- View ROC curves comparison
- See model comparison boxplots
- Results saved to SQLite database

---

## In Case of Errors

### Error 1: "ModuleNotFoundError: sklearn.ensemble._gb_losses"
```powershell
python train_models.py  # Retrain models
python app.py           # Try again
```

### Error 2: "Port 5000 already in use"
Edit app.py line ~130:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Error 3: "Module not found" (missing pandas, numpy, etc.)
```powershell
pip install -r requirements.txt
```

---

## Project Files to Know

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application (run this!) |
| `train_models.py` | Train ML models (run first time!) |
| `requirements.txt` | List of dependencies |
| `app/services/predictor.py` | Prediction logic |
| `app/database/db.py` | Database operations |
| `app/templates/index.html` | Web interface |
| `app/data/german_credit_data.csv` | Training dataset (2000 records) |

---

## Key Metrics

**Random Forest Model:**
- Training: 1500 samples, Testing: 500 samples
- Accuracy: 95%, Precision: 93%, Recall: 90%, F1: 92%

**Gradient Boosting Model:**
- Same train/test split
- Accuracy: 84%, Precision: 79%, Recall: 62%, F1: 69%

---

## Database
- SQLite database auto-created at `credit_results.db`
- Stores all predictions made through the web interface
- Clean data with timestamps

---

## Ports & Access
- **Local Machine**: http://127.0.0.1:5000
- **Network Access**: http://10.61.138.147:5000
- **Health Check**: http://127.0.0.1:5000/health
- **API Endpoint**: POST http://127.0.0.1:5000/api/predict/credit

---

## File Locations in Project
```
explainable-finance-ai/
├── app.py                    ← RUN THIS
├── train_models.py           ← RUN THIS FIRST
├── requirements.txt          ← pip install -r requirements.txt
├── venv/                     ← Virtual environment
├── app/
│   ├── models/              ← Trained .pkl files (auto-generated)
│   ├── data/                ← Dataset
│   ├── database/            ← SQLite db
│   ├── services/            ← Prediction logic
│   └── templates/           ← HTML/CSS
└── SETUP_INSTRUCTIONS.md    ← Full documentation
```

---

## Stopping the Server
Press: `CTRL+C` in the terminal

## Restarting
Flask debug mode auto-reloads on code changes. For full restart:
1. `CTRL+C` to stop
2. `python app.py` to restart
