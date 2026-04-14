"""
================================================================================
EXPLAINABLE FINANCE AI - Flask Web Application
================================================================================

PROJECT PURPOSE:
  Provides credit risk prediction using Machine Learning models with explainable
  AI (SHAP) visualizations. Users can input credit parameters and get predictions
  from both Random Forest and Gradient Boosting models.

HOW TO RUN THIS PROJECT:
  
  1. FIRST TIME SETUP (one-time):
     - Activate virtual environment: venv/Scripts/Activate.ps1
     - Install dependencies: pip install -r requirements.txt
     - Train models: python train_models.py
  
  2. RUN THE APPLICATION:
     - From project root: python app.py
     - Open browser: http://127.0.0.1:5000
     - Server runs on http://0.0.0.0:5000 (all interfaces, port 5000)
  
  3. IMPORTANT - Model Compatibility:
     If you see "ModuleNotFoundError: No module named 'sklearn.ensemble._gb_losses'":
     → Always run 'python train_models.py' to retrain models with current versions
  
APPLICATION STRUCTURE:
  - Main entry point: if __name__ == '__main__' (bottom of file)
  - Web interface: GET '/' returns index.html
  - API endpoint: POST '/api/predict/credit' returns prediction JSON
  - Health check: GET '/health' for monitoring

DEPENDENCIES:
  Flask, pandas, numpy, scikit-learn, joblib, SHAP, SQLAlchemy
  See requirements.txt for complete list

EXPECTED OUTPUT WHEN RUNNING:
  * Serving Flask app 'app'
  * Debug mode: on
  * Running on http://127.0.0.1:5000
  * Press CTRL+C to quit

================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os

from app.services.predictor import predict_both_models, _validate_credit_payload
from app.database.db import save_result, init_db

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder='app/templates')
init_db()


def load_evaluation_report():
    report_path = os.path.join(PROJECT_ROOT, 'model_evaluation_report.txt')
    metrics = {
        'random_forest': {},
        'gradient_boosting': {}
    }
    if not os.path.exists(report_path):
        return metrics

    current = None
    with open(report_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if 'RANDOM FOREST' in line:
                current = 'random_forest'
                continue
            if 'GRADIENT BOOSTING' in line:
                current = 'gradient_boosting'
                continue
            if current and line.startswith('•'):
                parts = line.replace('•', '', 1).split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    if key in {'accuracy', 'precision', 'recall', 'f1_score'}:
                        metrics[current][key] = value
    return metrics


@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(PROJECT_ROOT, filename)


@app.route('/')
def index():
    evaluation_metrics = load_evaluation_report()
    graph_files = [
        filename
        for filename in ['confusion_matrices.png', 'roc_curves.png', 'model_comparison_boxplots.png']
        if os.path.exists(os.path.join(PROJECT_ROOT, filename))
    ]
    return render_template('index.html', evaluation_metrics=evaluation_metrics, graph_files=graph_files)


@app.route('/api/predict/credit', methods=['POST'])
def predict_credit():
    try:
        data = request.json
        payload = {
            'Age': float(data.get('age', 30)),
            'Job': int(data.get('job', 1)),
            'Credit amount': float(data.get('loan_amount', 10000)),
            'Duration': float(data.get('duration', 12)),
            'Sex': str(data.get('sex', 'male')),
            'Housing': str(data.get('housing', 'own')),
            'Saving accounts': str(data.get('saving_accounts', 'little')),
            'Checking account': str(data.get('checking_account', 'moderate')),
            'Purpose': str(data.get('purpose', 'education')),
        }

        _validate_credit_payload(payload)
        results = predict_both_models(payload)

        save_result({
            'age': float(data.get('age', 30)),
            'rf_prob': results['random_forest']['probability'],
            'gb_prob': results['gradient_boosting']['probability']
        }, 'credit_results')

        evaluation_metrics = load_evaluation_report()

        return jsonify({
            'success': True,
            'random_forest': results['random_forest'],
            'gradient_boosting': results['gradient_boosting'],
            'threshold': 0.5,
            'evaluation_metrics': evaluation_metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
