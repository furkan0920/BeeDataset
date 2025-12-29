import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import warnings

warnings.filterwarnings("ignore")

def grafik_olustur_profesyonel():
    
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

    epochs = 50
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1, 
                          warm_start=True, random_state=42, solver='adam')

    history = {
        'train_loss': [], 'val_loss': [],
        'precision': [], 'recall': [], 'f1': [], 'accuracy': []
    }

    for epoch in range(epochs):
        model.fit(X_train, y_train) 
        
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        history['train_loss'].append(model.loss_)
        history['val_loss'].append(log_loss(y_test, y_prob)) 
        
        history['precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        history['recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        history['f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        history['accuracy'].append(accuracy_score(y_test, y_pred))


    sns.set_style("whitegrid")
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = range(1, epochs + 1)

    axs[0, 0].plot(epochs_range, history['train_loss'], label='Train Loss', color='#d62728', linewidth=2.5) # Kırmızı
    axs[0, 0].plot(epochs_range, history['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2) # Turuncu
    axs[0, 0].set_title('Loss (Kayıp) Fonksiyonu')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss Değeri')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs_range, history['accuracy'], label='Accuracy', color='#1f77b4', linewidth=2.5) # Mavi
    axs[0, 1].set_title('Accuracy (Doğruluk) Değişimi')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs_range, history['precision'], label='Precision', color='#2ca02c', linewidth=2) # Yeşil
    axs[1, 0].plot(epochs_range, history['recall'], label='Recall', color='#9467bd', linestyle='--', linewidth=2) # Mor
    axs[1, 0].set_title('Precision ve Recall Analizi')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs_range, history['f1'], label='F1-Score', color='#8c564b', linewidth=2.5) 
    axs[1, 1].set_title('F1-Score (Dengeli Başarı) Gelişimi')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()

    plt.suptitle('Bee Dataset: Model Eğitim Performansı ve Metrik Analizi', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig('bee_performance_graphs.png', dpi=300) 
    plt.show()

if __name__ == "__main__":
    grafik_olustur_profesyonel()