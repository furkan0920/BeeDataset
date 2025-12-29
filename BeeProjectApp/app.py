from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

print("Sistem yükleniyor...")
data = joblib.load('ari_saglik_sistemi.pkl')
model = data['model']
scaler = data['scaler']
le_dict = data['le_dict']
le_target = data['le_target']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    confidence = None
    alert_type = None

    options = {
        'subspecies': le_dict['subspecies'].classes_,
        'caste': le_dict['caste'].classes_,
        'pollen': le_dict['pollen_carrying'].classes_
    }

    if request.method == 'POST':
        try:
            subspecies = request.form['subspecies']
            caste = request.form['caste']
            pollen = request.form['pollen']

            val_sub = le_dict['subspecies'].transform([subspecies])[0]
            val_caste = le_dict['caste'].transform([caste])[0]
            val_pollen = le_dict['pollen_carrying'].transform([pollen])[0]

            input_data = np.array([[val_sub, val_caste, val_pollen]])
            input_scaled = scaler.transform(input_data)
            
            pred_idx = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled).max()
            
            result = le_target.inverse_transform([pred_idx])[0]
            
            prediction_text = result.upper()
            confidence = f"%{proba*100:.2f}"
            
            if result == 'healthy':
                alert_type = 'success' 
            else:
                alert_type = 'danger'  

        except Exception as e:
            prediction_text = f"Hata oluştu: {e}"

    return render_template('index.html', 
                           options=options, 
                           prediction=prediction_text, 
                           confidence=confidence,
                           alert_type=alert_type)

if __name__ == '__main__':
    app.run(debug=True)