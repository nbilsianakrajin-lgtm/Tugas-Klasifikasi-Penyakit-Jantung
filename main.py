# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler # BARU: Untuk menyamakan skala data
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # BARU: Untuk laporan detail

print("--- Memulai Proyek Klasifikasi Penyakit Jantung ---")

# 2. MEMBACA DATA
df = pd.read_csv('heart.csv')
print(f"Total data pasien: {len(df)} baris")

# 3. PEMISAHAN FITUR (X) DAN TARGET (Y)
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# 4. PRA-PEMROSESAN DATA
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"Jumlah fitur setelah diubah ke angka: {X_encoded.shape[1]} kolom")

# 5. MEMBAGI DATA (TRAIN & TEST)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- UPGRADE: SCALING DATA ---
# Menyamakan skala fitur numerik agar model AI lebih mudah belajar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MELATIH MODEL (RANDOM FOREST dengan Parameter Lebih Baik)
print("Sedang melatih model AI...")
# Kita batasi kedalaman pohon (max_depth) agar AI tidak terlalu 'menghafal' data
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. PENGUJIAN MODEL
y_pred = model.predict(X_test_scaled)
akurasi = accuracy_score(y_test, y_pred)
print(f"\n=== HASIL EVALUASI ===")
print(f"Akurasi Model: {akurasi * 100:.2f}%")

# --- UPGRADE: CLASSIFICATION REPORT (Seperti punya teman Anda) ---
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Sehat (0)', 'Sakit (1)']))

# 8. MEMBUAT GAMBAR 1: CONFUSION MATRIX
print("Membuat gambar confusion_matrix.png...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Sehat', 'Prediksi Sakit'], 
            yticklabels=['Aktual Sehat', 'Aktual Sakit'])
plt.title('Confusion Matrix - Klasifikasi Penyakit Jantung')
plt.ylabel('Kondisi Sebenarnya')
plt.xlabel('Hasil Prediksi AI')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 9. MEMBUAT GAMBAR 2: FEATURE IMPORTANCE
print("Membuat gambar feature_importance.png...")
importances = model.feature_importances_
feature_names = X_encoded.columns

feature_importance_df = pd.DataFrame({
    'Fitur': feature_names,
    'Tingkat Kepentingan': importances
}).sort_values(by='Tingkat Kepentingan', ascending=False)

top_15_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 8))
# Memperbaiki peringatan (FutureWarning) dari Seaborn
sns.barplot(x='Tingkat Kepentingan', y='Fitur', data=top_15_features, palette='viridis', hue='Fitur', legend=False)
plt.title('15 Fitur Paling Berpengaruh terhadap Penyakit Jantung')
plt.xlabel('Skor Kepentingan')
plt.ylabel('Atribut Pasien')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\n--- Selesai! Silakan cek folder Anda. ---")