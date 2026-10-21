# Tugas-Klasifikasi-Penyakit-Jantung

# Klasifikasi Penyakit Jantung (Heart Disease Classification)

Proyek ini adalah tugas mata kuliah Big Data yang bertujuan untuk memprediksi kemungkinan seseorang menderita penyakit jantung berdasarkan atribut data pasien (seperti usia, jenis kelamin, BMI, tekanan darah, kolesterol, dll).

# Dataset
* Dataset yang digunakan: `heart.csv` (100.000 baris data).
* Terdiri dari 26 fitur (numerik, kategorikal, dan biner).

# Alur Kerja Program
1. **Pra-pemrosesan Data**: Mengubah data teks (seperti "Male", "Northeast") menjadi bentuk numerik menggunakan *One-Hot Encoding*.
2. **Pemodelan**: Menggunakan algoritma *Random Forest Classifier* untuk melatih data.
3. **Evaluasi**: Menghitung akurasi model dan menampilkan *Confusion Matrix*.

## 📈 Hasil
* **Akurasi Model:** 75.63%
* Visualisasi hasil dapat dilihat pada file gambar di repositori ini:
  * `confusion_matrix.png`
  * `feature_importance.png`

## 🚀 Cara Menjalankan Program
1. Pastikan library terinstal: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Jalankan file utama: `python main.py`
