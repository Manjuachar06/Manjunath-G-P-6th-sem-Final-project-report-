from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import os
import joblib
import pandas as pd
import numpy as np
from utils import extract_text
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

MODEL_FILE = 'model.pkl'

# Load the trained model globally
try:
    vectorizer = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model not found or failed to load. Please run train_model.py first! ({e})")
    vectorizer = None

# ============================================================================
# LOAD RESUME SCORING MODELS (ML Models for scoring)
# ============================================================================
print("\n" + "="*60)
print("Loading Resume Scoring Models...")
print("="*60)

ridge_model = None
rf_model = None
scaler = None

try:
    ridge_model = joblib.load('ml_results/ridge_regression_model.pkl')
    print("Ridge Regression model loaded")
except Exception as e:
    print(f"Ridge model not found. Run: python train_resume_models.py ({e})")

try:
    rf_model = joblib.load('ml_results/random_forest_regressor_model.pkl')
    print("Random Forest model loaded")
except Exception as e:
    print(f"Random Forest model not found. Run: python train_resume_models.py ({e})")

try:
    scaler = joblib.load('ml_results/scaler.pkl')
    print("Feature scaler loaded")
except Exception as e:
    print(f"Scaler not found ({e})")

MODELS_READY = ridge_model is not None and rf_model is not None and scaler is not None
if MODELS_READY:
    print("\nResume scoring models are READY")
else:
    print("\nResume scoring models NOT ready")

print("="*60)

# Function to extract features from resume text
def extract_resume_features(resume_text, requirements_text=""):
    """Extract numerical features from resume text for ML scoring"""
    import re
    try:
        word_count = len(resume_text.split())
        sentence_count = max(len(resume_text.split('.')) - 1, 1)
        
        features = {
            'word_count': min(max(word_count, 100), 800),
            'avg_sentence_length': min(max(word_count / sentence_count, 5), 25),
            'keyword_relevance_score': 0.5,  # Default, overwritten below
            'education_level': 2,  # Default bachelor's degree
            'years_experience': 3,  # Default 3 years
            'skills_count': min(max(resume_text.lower().count('skill') + resume_text.lower().count('competency'), 1), 25),
            'certification_count': min(max(resume_text.lower().count('certif') + resume_text.lower().count('certified'), 0), 10),
            'project_count': min(max(resume_text.lower().count('project'), 0), 20),
            'grammar_score': 0.85,  # Default good grammar
            'formatting_score': 0.8,  # Default good formatting
            'has_gpa': 1 if 'gpa' in resume_text.lower() else 0,
            'has_linkedin': 1 if 'linkedin' in resume_text.lower() else 0,
            'has_portfolio': 1 if 'portfolio' in resume_text.lower() or 'github' in resume_text.lower() else 0,
        }
        
        # Calculate real keyword relevance
        if requirements_text and vectorizer:
            try:
                tfidf_matrix = vectorizer.transform([requirements_text, resume_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                features['keyword_relevance_score'] = min(max(similarity, 0.1), 0.95)
            except:
                pass
                
        # Detect education level from keywords
        if any(term in resume_text.lower() for term in ['phd', 'doctorate']):
            features['education_level'] = 4
        elif any(term in resume_text.lower() for term in ['master', 'mba', 'm.sc']):
            features['education_level'] = 3
        elif any(term in resume_text.lower() for term in ['bachelor', 'b.s', 'b.a']):
            features['education_level'] = 2
        else:
            features['education_level'] = 1
        
        # Better years of experience detection
        exp_match = re.search(r'(\d+)\+?\s*(years|yrs)\s*(of)?\s*experience', resume_text.lower())
        if exp_match:
            features['years_experience'] = min(float(exp_match.group(1)), 30)
        else:
            features['years_experience'] = 3
        
        return features
    except:
        # Return default features if extraction fails
        return {
            'word_count': 300, 'avg_sentence_length': 15, 'keyword_relevance_score': 0.5,
            'education_level': 2, 'years_experience': 3, 'skills_count': 10,
            'certification_count': 2, 'project_count': 3, 'grammar_score': 0.8,
            'formatting_score': 0.8, 'has_gpa': 0, 'has_linkedin': 0, 'has_portfolio': 0
        }

def score_resume_with_ml(resume_text, requirements_text=""):
    """Score a resume using trained ML models"""
    if not MODELS_READY:
        return {'error': 'Models not trained. Run: python train_resume_models.py'}
    
    try:
        # Extract features
        features = extract_resume_features(resume_text, requirements_text)
        
        # Create DataFrame with exact column names in correct order
        feature_names = ['word_count', 'avg_sentence_length', 'keyword_relevance_score',
                         'education_level', 'years_experience', 'skills_count',
                         'certification_count', 'project_count', 'grammar_score',
                         'formatting_score', 'has_gpa', 'has_linkedin', 'has_portfolio']
        
        X = pd.DataFrame([features], columns=feature_names)
        
        # Scale features for Ridge model
        X_scaled = scaler.transform(X)
        
        # Get predictions
        ridge_score = float(ridge_model.predict(X_scaled)[0])
        rf_score = float(rf_model.predict(X)[0])
        
        # Clip scores to 0-100 range
        ridge_score = np.clip(ridge_score, 0, 100)
        rf_score = np.clip(rf_score, 0, 100)
        
        # Average of both models
        avg_score = (ridge_score + rf_score) / 2
        
        return {
            'ridge_regression_score': round(ridge_score, 2),
            'random_forest_score': round(rf_score, 2),
            'average_score': round(avg_score, 2),
            'features': features
        }
    except Exception as e:
        return {'error': f'Error scoring resume: {str(e)}'}

def get_match_score(requirements_text, resume_text):
    if not vectorizer:
        raise ValueError("Model is not loaded.")
        
    # Transform texts using the trained vectorizer
    # transform takes an iterable of strings
    tfidf_matrix = vectorizer.transform([requirements_text, resume_text])
    
    # Calculate cosine similarity between requirements (index 0) and resume (index 1)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage
    score_percentage = round(similarity * 100, 2)
    return score_percentage

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        requirements = request.form.get('requirements', '')
        if not requirements.strip():
            flash("Please enter company requirements.")
            return redirect(request.url)
            
        if 'resume' not in request.files:
            flash('No file uploaded.')
            return redirect(request.url)
            
        file = request.files['resume']
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)
            
        if file:
            try:
                # Extract text from the uploaded file
                resume_text = extract_text(file, file.filename)
                
                if not resume_text.strip():
                    flash("Could not extract any text from the provided file.")
                    return redirect(request.url)
                
                # Validate resume has meaningful content (minimum 20 words)
                word_count = len(resume_text.split())
                if word_count < 20:
                    flash("Resume content is too short or invalid. Please upload a valid resume with at least 20 words.")
                    return redirect(request.url)
                
                # Get ML scores (NEW)
                ml_scores = score_resume_with_ml(resume_text, requirements)
                
                # Get matching score (OLD)
                match_score = None
                if vectorizer:
                    try:
                        match_score = get_match_score(requirements, resume_text)
                    except:
                        pass
                
                return render_template('index.html', 
                                     ml_scores=ml_scores,
                                     match_score=match_score,
                                     requirements=requirements)
            
            except Exception as e:
                flash(f"An error occurred during parsing: {str(e)}")
                return redirect(request.url)

    return render_template('index.html', ml_scores=None, match_score=None, requirements="")

@app.route('/ml-results/<filename>')
def ml_results(filename):
    """Serve ML result image/CSV files from the ml_results folder."""
    if '..' in filename or '/' in filename:
        flash("Invalid file request", category='error')
        return redirect(url_for('index'))

    file_path = os.path.join('ml_results', filename)
    if not os.path.exists(file_path):
        flash(f"File not found: {filename}", category='error')
        return redirect(url_for('index'))

    if filename.endswith('.png'):
        return send_file(file_path, mimetype='image/png')
    elif filename.endswith('.csv'):
        return send_file(file_path, mimetype='text/csv', as_attachment=True, download_name=filename)
    else:
        flash("Invalid file type", category='error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
