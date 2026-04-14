# AI Resume Analyzer with ML Scoring Models

## Project Overview
A Flask-based AI resume analysis system that uses **2 trained ML models** to score and evaluate resumes based on resume quality and features, trained on 500 resume samples.

### Two ML Models
1. **Ridge Regression** - R² Score: 0.8339 (Better performance)
2. **Random Forest** - Captures feature importance and non-linear patterns

---

## Features

### ✅ Main Features
- **Resume Scoring with 2 ML Models**: Get predictions from both Ridge Regression and Random Forest
- **Resume Quality Assessment**: Scores 0-100 based on:
  - Word count & structure
  - Keywords & relevance
  - Education level
  - Work experience
  - Skills count
  - Certifications
  - Grammar & formatting
  - Portfolio/GitHub presence

- **Comprehensive Metrics**:
  - **R² Score**: Model prediction accuracy
  - **RMSE**: Root Mean Squared Error
  - **MAE**: Mean Absolute Error
  - **MAPE**: Mean Absolute Percentage Error

- **7 Visualizations**:
  - Model Performance Comparison (bar chart)
  - Actual vs Predicted Scores (scatter plots)
  - Residuals Analysis
  - R² Comparison
  - Error Metrics (MAE vs RMSE)
  - Feature Importance
  - Score Distribution

---

## Project Structure

```
c:\Users\m9453\OneDrive\Desktop\Ai resuma
├── app.py                        # Flask application
├── train_resume_models.py       # ML training script
├── utils.py                      # Utility functions
├── requirements.txt              # Python dependencies
├── static/
│   └── style.css                # CSS stylesheet
├── templates/
│   └── index.html               # Resume analyzer page
├── uploads/                      # Uploaded resume files
└── ml_results/                   # ML results directory
    ├── resume_dataset.csv       # Training dataset (500 samples)
    ├── model_comparison.csv     # Performance metrics
    ├── ridge_regression_model.pkl
    ├── random_forest_regressor_model.pkl
    ├── scaler.pkl
    └── *.png                     # 7 visualization graphs
```

---

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train ML Models (One-time setup)
```bash
python train_resume_models.py
```

**Output:**
```
==============================================================================
RESUME SCORING DATASET & MODEL TRAINING
==============================================================================

✓ Dataset created: 500 resume samples
✓ Training set: 400 samples  
✓ Test set: 100 samples

Ridge Regression:
  • R² Score:   0.8339
  • RMSE:       5.0728
  • MAE:        4.1462

Random Forest:
  • R² Score:   0.6801
  • RMSE:       7.0402
  • MAE:        5.7440

✅ All 12 files generated in ml_results/
```

### Step 3: Run Flask Application
```bash
python app.py
```

**Output:**
```
Model loaded successfully.
✓ Ridge Regression model loaded
✓ Random Forest model loaded
✓ Feature scaler loaded
✅ Resume scoring models are READY
```

Open browser: **http://localhost:5000**

---

## How to Use

### 1. Analyze Resume with ML Models
- Go to home page
- Upload a resume (.pdf, .docx, .txt)
- Click "Analyze Resume"
- Get scores from both models:
  - **Ridge Regression Score** (higher accuracy)
  - **Random Forest Score** (feature importance)
  - **Average Score** (combined prediction)

### 2. View ML Model Comparison
- Click "📊 View ML Models Comparison"
- See:
  - Performance metrics table
  - 7 detailed visualizations
  - Download options for data

### 3. Download Results
- Resume dataset (CSV)
- Model comparison metrics (CSV)
- All visualizations (PNG)

---

## Model Details

### Dataset (resume_dataset.csv)
- **500 resume samples** generated with realistic features
- **Features extracted:**
  - Word count (100-800)
  - Average sentence length  
  - Keyword relevance score
  - Education level (1-4)
  - Years of experience (0-30)
  - Skills count
  - Certifications
  - Projects
  - Grammar score
  - Formatting score
  - LinkedIn presence
  - Portfolio/GitHub presence

### Ridge Regression Model ⭐ (Better)
- **Type**: Linear regression with L2 regularization
- **Advantages**: 
  - Highest R² = 0.8339
  - Lower RMSE = 5.07
  - Better generalization
- **Best for**: Accurate resume scoring

### Random Forest Model
- **Type**: Ensemble of 100 decision trees
- **Advantages**:
  - Non-linear pattern capture
  - Feature importance analysis
  - Robust to outliers
- **Use case**: Understanding which features matter most

---

## Performance Metrics Explained

### R² Score (Coefficient of Determination)
- **Range**: 0 to 1
- **Meaning**: How well the model explains variance in data
- **Ridge: 0.8339** = Model explains 83.39% of score variation
- **RF: 0.6801** = Model explains 68.01% of score variation

### RMSE (Root Mean Squared Error)
- **Lower is better**
- Average prediction error in points
- **Ridge: 5.07** = Average error of ±5 points
- **RF: 7.04** = Average error of ±7 points

### MAE (Mean Absolute Error)  
- **Lower is better**
- Average absolute error
- **Ridge: 4.15** = Average error of ±4.15 points
- **RF: 5.74** = Average error of ±5.74 points

### MAPE (Mean Absolute Percentage Error)
- **Lower is better**
- Error as a percentage
- Easier to interpret for stakeholders

---

## Routes & Endpoints

```
GET  /                           Home page (upload resume)
POST /                           Submit resume for scoring
GET  /ml-models                  View model comparison
GET  /ml-results/<filename>      Download files (CSV, PNG)
```

---

## Generated Visualizations

### 01: Model Performance Comparison
Bar chart comparing all metrics (R², RMSE, MAE, MAPE) for both models

### 02: Actual vs Predicted
Scatter plots showing how well predictions match actual scores

### 03: Residuals Analysis  
Plot of prediction errors to identify systematic patterns

### 04: R² Comparison
Direct comparison of model accuracy (R² scores)

### 05: Error Metrics
MAE vs RMSE comparison for easy interpretation

### 06: Feature Importance
Random Forest's top 10 most important features for scoring

### 07: Score Distribution
Histogram showing actual vs predicted score distribution

---

## Customization

### Train on Your Own Dataset
Edit `train_resume_models.py`:

```python
# Replace the data creation section with:
df = pd.read_csv('your_resume_data.csv')
X = df.drop('target_score', axis=1)
y = df['target_score']
```

### Add More Features
```python
data = {
    'your_feature_1': values,
    'your_feature_2': values,
    # ... more features
}
```

### Try Different Models
```python
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'Ridge': Ridge(),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor()
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Models not trained" | Run: `python train_resume_models.py` |
| Module not found | Run: `pip install -r requirements.txt` |
| Port 5000 in use | Change port in app.py: `app.run(port=5001)` |
| Can't extract resume | Ensure file is .pdf, .docx, or .txt |
| Scores seem wrong | Train new models with better dataset |

---

## Model Training Time
- **Ridge Regression**: < 1 second
- **Random Forest**: < 2 seconds
- **Visualizations**: < 5 seconds
- **Total**: ~10 seconds

---

## Files Generated After Training

### Data Files
- `resume_dataset.csv` - 500 training samples
- `model_comparison.csv` - Metrics table for both models

### Model Files
- `ridge_regression_model.pkl` - Trained Ridge model
- `random_forest_regressor_model.pkl` - Trained RF model
- `scaler.pkl` - Feature scaler for ML preprocessing

### Visualizations (300 DPI PNG)
- `01_model_performance_comparison.png`
- `02_actual_vs_predicted.png`
- `03_residuals_analysis.png`
- `06_feature_importance.png`
- `07_score_distribution.png`

---

## Requirements

### Python Packages
```
Flask==3.0.0
scikit-learn==1.3.2
pandas==2.1.4
PyPDF2==3.0.1
python-docx==1.1.0
requests==2.31.0
joblib==1.3.2
werkzeug==3.0.1
matplotlib==3.8.2
seaborn==0.13.0
numpy==1.24.3
```

### System Requirements
- Python 3.8+
- 100 MB disk space
- 512 MB RAM minimum

---

## How Scoring Works

### Feature Extraction
When you upload a resume, the system:
1. Extracts text from PDF/DOCX/TXT
2. Analyzes 13 features
3. Scales features using fitted scaler
4. Feeds to both ML models

### Prediction
```
Input Resume
    ↓
Feature Extraction (13 features)
    ↓
Feature Scaling (Ridge only)
    ↓
Ridge Regression Model → Score 1 (0-100)
    ↓
Random Forest Model → Score 2 (0-100)
    ↓
Average Score = (Score 1 + Score 2) / 2
```

---

## Expected Scores

- **80-100**: Excellent resume (well-structured, experienced)
- **60-80**: Good resume (most features present)
- **40-60**: Average resume (some features missing)
- **20-40**: Needs work (missing key sections)
- **0-20**: Poor resume (minimal content)

---

## Example Output

```
📊 RESUME SCORING RESULTS

Ridge Regression Score:      85.32 / 100
Random Forest Score:         79.45 / 100
Average Score:               82.39 / 100

⭐ Excellent Resume Quality
```

---

## Testing the System

### With a Sample Resume
1. Create a text file with resume content:
   ```
   NAME: John Doe
   EDUCATION: Bachelor's Degree in Computer Science
   EXPERIENCE: 5 years as Software Engineer
   SKILLS: Python, JavaScript, SQL, Docker
   CERTIFICATIONS: AWS Certified Developer
   PORTFOLIO: github.com/johndoe
   ```

2. Upload and test

### Expected Results
- Resume with good structure: 70-90
- Resume with missing sections: 40-60
- Very minimal resume: 20-40

---

## Future Enhancements

- [ ] Support for more resume formats
- [ ] Advanced NLP for keyword extraction
- [ ] Batch resume analysis
- [ ] Custom scoring weights
- [ ] Resume improvement suggestions
- [ ] Comparison with job descriptions
- [ ] API endpoint for integration

---

## Support

For issues or questions:
1. Check that all dependencies are installed
2. Ensure models are trained (`python train_resume_models.py`)
3. Check Flask app output for error messages
4. Verify file formats are supported

---

**Created**: April 2026  
**Framework**: Flask + Scikit-learn + Pandas + Matplotlib  
**Models**: Ridge Regression + Random Forest  
**Dataset**: 500 synthetic resume samples

