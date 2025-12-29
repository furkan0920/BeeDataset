import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle
import warnings

warnings.filterwarnings("ignore")

def confusion_ve_roc_ciz():
    print("⏳ Grafikler hazırlanıyor... (Lütfen bekleyin)")
    
    # --- 1. VERİ HAZIRLAMA ---
    df = pd.read_csv("bee_data.csv")
    y = df['health']
    
    # Gerçekçi Senaryo: Sadece biyolojik veriler
    X = df[['subspecies', 'caste', 'pollen_carrying']].copy()

    # Encoder
    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    
    y_encoded = le.fit_transform(y)
    classes = le.classes_ # Sınıf isimleri
    n_classes = len(classes)
    
    # Ölçekleme
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Eğitim/Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 2. MODEL EĞİTİMİ (MLP) ---
    # ROC çizimi için 'probability' (olasılık) değerlerine ihtiyacımız var
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) # Olasılıklar (ROC için şart)

    # --- 3. GRAFİK ÇİZİMİ (YAN YANA 2 GRAFİK) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- A. CONFUSION MATRIX (SOL) ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14)
    ax1.set_ylabel('Gerçek Sınıflar')
    ax1.set_xlabel('Tahmin Edilen Sınıflar')
    ax1.tick_params(axis='x', rotation=45)

    # --- B. ROC EĞRİLERİ (SAĞ) ---
    # Hedef veriyi binarize et (Çoklu sınıf ROC için şarttır)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    # Her sınıf için FPR, TPR ve AUC hesapla
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Renk döngüsü
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown'])
    
    for i, color in zip(range(n_classes), colors):
        ax2.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))

    ax2.plot([0, 1], [0, 1], 'k--', lw=2) # Rastgele tahmin çizgisi (Diyagonal)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    ax2.set_ylabel('True Positive Rate (Duyarlılık)')
    ax2.set_title('Çok Sınıflı ROC Analizi', fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('confusion_roc_analysis.png')
    print("✅ Grafik 'confusion_roc_analysis.png' olarak kaydedildi ve ekrana getiriliyor.")
    plt.show()

if __name__ == "__main__":
    confusion_ve_roc_ciz()