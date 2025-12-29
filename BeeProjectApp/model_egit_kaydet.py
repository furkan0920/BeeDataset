import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

print("⏳ Model eğitiliyor... Lütfen bekleyin.")

# 1. Veri Hazırlama
df = pd.read_csv("bee_data.csv")
y = df['health']
X = df[['subspecies', 'caste', 'pollen_carrying']].copy()

# Encoders (Flask arayüzündeki seçenekler için bunları saklamalıyız)
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Hedef değişken
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Model Eğitimi (MLP - Sinir Ağı)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_scaled, y_encoded)

# 3. Kaydetme
data = {
    'model': model,
    'scaler': scaler,
    'le_dict': le_dict,
    'le_target': le_target
}

joblib.dump(data, 'ari_saglik_sistemi.pkl')
print("✅ Başarılı! 'ari_saglik_sistemi.pkl' dosyası oluşturuldu.")