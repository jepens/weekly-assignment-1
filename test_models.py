# Import library yang diperlukan
import pickle  # Untuk membaca file model yang tersimpan
import pandas as pd  # Untuk manipulasi data dalam bentuk DataFrame

# Memulai proses loading model dan scaler
print("Memuat model dan scaler...")

# Membaca model Linear Regression dari file
with open('boston_house_lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)
print("Model Linear Regression berhasil dimuat")

# Membaca model Random Forest dari file
with open('boston_house_rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
print("Model Random Forest berhasil dimuat")

# Membaca StandardScaler dari file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
print("Scaler berhasil dimuat")

# Mendefinisikan nama-nama fitur sesuai dengan dataset Boston Housing
feature_names = [
    'crim',     # Tingkat kriminalitas
    'zn',       # Proporsi lahan perumahan
    'indus',    # Proporsi industri
    'chas',     # Dekat dengan Charles River (0 atau 1)
    'nox',      # Konsentrasi oksida nitrat
    'rm',       # Rata-rata jumlah kamar
    'age',      # Proporsi unit yang dibangun sebelum 1940
    'dis',      # Jarak ke pusat kota
    'rad',      # Indeks aksesibilitas jalan raya
    'tax',      # Pajak properti
    'ptratio',  # Rasio murid-guru
    'black',    # Proporsi penduduk kulit hitam
    'lstat'     # Status ekonomi penduduk
]

# Membuat data sampel untuk testing
# Data ini adalah contoh satu rumah dengan nilai-nilai fiturnya
sample_data = pd.DataFrame([[
    0.00632,  # crim
    18.0,     # zn
    2.31,     # indus
    0,        # chas
    0.538,    # nox
    6.575,    # rm
    65.2,     # age
    4.09,     # dis
    1,        # rad
    296,      # tax
    15.3,     # ptratio
    396.9,    # black
    4.98      # lstat
]], columns=feature_names)

# Melakukan standardisasi data menggunakan scaler yang sudah dilatih
sample_scaled = scaler.transform(sample_data)

# Membuat prediksi menggunakan kedua model
# Prediksi dengan Linear Regression
lr_pred = lr_model.predict(sample_scaled)
# Prediksi dengan Random Forest
rf_pred = rf_model.predict(sample_scaled)

# Menampilkan hasil prediksi
print("\nHasil Prediksi untuk Data Sampel:")
print(f"Prediksi Linear Regression: ${lr_pred[0]:,.2f}k")
print(f"Prediksi Random Forest: ${rf_pred[0]:,.2f}k")

# Menambahkan analisis perbandingan prediksi
print("\nAnalisis Hasil Prediksi:")
perbedaan = abs(lr_pred[0] - rf_pred[0])
print(f"Perbedaan antara kedua model: ${perbedaan:.2f}k")
print(f"Rata-rata prediksi: ${(lr_pred[0] + rf_pred[0])/2:.2f}k")

# Menampilkan interpretasi hasil
print("\nInterpretasi:")
if perbedaan < 5:
    print("Kedua model memberikan prediksi yang relatif konsisten (perbedaan < $5k)")
else:
    print("Terdapat perbedaan yang cukup signifikan antara prediksi kedua model (perbedaan > $5k)")