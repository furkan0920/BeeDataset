import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

global y_test, y_pred, y_prob, classes, y_encoded

def score():
    
    global y_test, y_pred, y_prob, classes, y_encoded
    
    
    df = pd.read_csv("bee_data.csv")
    y = df['health']
    
    X = df[['subspecies', 'caste', 'pollen_carrying']].copy()

    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col])

    y_encoded = le.fit_transform(y)
    classes = le.classes_

    X_train, X_test, y_train, y_test_local = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_test = y_test_local
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    

def f1score():
    score = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f" F1-Score: {score:.4f}")

def get_precision():
    score = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Precision: {score:.4f}")

def get_recall():
    score = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f" Recall: {score:.4f}")

def get_mAP():
    y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
    score = average_precision_score(y_test_bin, y_prob, average="macro")

def show_confusion_matrix():
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Gerçekçi Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    plt.show()

if __name__ == "__main__":
    score()
    
    get_precision()
    get_recall()
    f1score()
    get_mAP()
    
    show_confusion_matrix()