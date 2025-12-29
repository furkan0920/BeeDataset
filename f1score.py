import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import warnings


def tablolari_olustur():
    
    df = pd.read_csv("bee_data.csv")
    y = df['health']
    
    X = df[['subspecies', 'caste', 'pollen_carrying']].copy()

    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    
    y_encoded = le.fit_transform(y)
    classes = le.classes_ 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                          random_state=42, solver='adam')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(4)
    df_classes = df_report.iloc[:-3, :] 
    df_summary = df_report.iloc[-3:, :]

    print("\n--- TABLO 4.3.1: SINIF BAZLI PERFORMANS METRİKLERİ ---")
    print(df_classes.to_string())
    
    print("\n\n--- TABLO 4.3.2: GENEL MODEL ÖZETİ ---")
    print(df_summary.to_string())

    df_report.to_csv("performans_tablosu.csv")

if __name__ == "__main__":
    tablolari_olustur()