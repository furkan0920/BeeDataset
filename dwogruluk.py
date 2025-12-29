import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def accuracy_tablosu_olustur():
    
    df = pd.read_csv("bee_data.csv")
    y = df['health']
    
    X = df[['subspecies', 'caste', 'pollen_carrying']].copy()

    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                          random_state=42, solver='adam')

    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print("Çapraz doğrulama (Cross-Validation) yapılıyor...")
    val_scores = cross_val_score(model, X_train, y_train, cv=5)
    val_acc = val_scores.mean()

    results = {
        'Veri Kümesi (Dataset)': ['Eğitim (Training)', 'Doğrulama (Validation)', 'Test (Testing)'],
        'Doğruluk Oranı (Accuracy)': [train_acc, val_acc, test_acc],
        'Açıklama': [
            'Modelin öğrendiği veri üzerindeki başarısı', 
            '5-Fold Cross Validation ortalaması', 
            'Modelin hiç görmediği veri üzerindeki başarısı'
        ]
    }
    
    df_results = pd.DataFrame(results)
    
    df_results['Doğruluk Oranı (Accuracy)'] = df_results['Doğruluk Oranı (Accuracy)'].apply(lambda x: f"%{x*100:.2f}")

    print(df_results.to_string(index=False))
    
    df_results.to_csv("accuracy_tablosu.csv", index=False)

if __name__ == "__main__":
    accuracy_tablosu_olustur()