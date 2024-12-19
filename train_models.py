# Import library-library yang diperlukan untuk pemrosesan data dan machine learning
import pandas as pd  # Untuk manipulasi dan analisis data
import numpy as np   # Untuk operasi numerik dan array
from sklearn.model_selection import train_test_split  # Untuk membagi dataset
from sklearn.preprocessing import StandardScaler     # Untuk standardisasi fitur
from sklearn.linear_model import LinearRegression   # Model regresi linear
from sklearn.ensemble import RandomForestRegressor  # Model random forest
import pickle  # Untuk menyimpan model ke file

# Mendefinisikan nama-nama fitur untuk dokumentasi
feature_names = [
    'crim',     # Tingkat kriminalitas per kapita
    'zn',       # Proporsi lahan perumahan untuk lot besar
    'indus',    # Proporsi area bisnis non-retail
    'chas',     # Variabel dummy Charles River (1 jika dekat, 0 jika tidak)
    'nox',      # Konsentrasi oksida nitrat
    'rm',       # Rata-rata jumlah kamar per hunian
    'age',      # Proporsi unit yang dibangun sebelum 1940
    'dis',      # Jarak tertimbang ke pusat kerja di Boston
    'rad',      # Indeks aksesibilitas ke jalan raya radial
    'tax',      # Nilai pajak properti
    'ptratio',  # Rasio murid-guru
    'black',    # Proporsi penduduk kulit hitam
    'lstat'     # Persentase populasi dengan status lebih rendah
]

# Membaca dataset dari file CSV
print("Memuat dataset...")
df = pd.read_csv('dataset.csv')

# Tahap preprocessing data
print("Melakukan preprocessing data...")
# Memisahkan fitur (X) dan target (y)
X = df.drop(['ID', 'medv'], axis=1)  # Menghapus kolom ID dan target
y = df['medv']  # Target adalah median value dari rumah dalam ribuan dollar

# Membagi dataset menjadi data training dan testing
print("Membagi data menjadi data training dan testing...")
# test_size=0.2 artinya 20% untuk testing, 80% untuk training
# random_state=42 untuk memastikan hasil yang konsisten setiap kali dijalankan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melakukan standardisasi fitur
print("Melakukan standardisasi fitur...")
scaler = StandardScaler()  # Inisialisasi scaler
# Fit dan transform data training
X_train_scaled = scaler.fit_transform(X_train)
# Transform data testing menggunakan parameter dari data training
X_test_scaled = scaler.transform(X_test)

# Melatih model Linear Regression
print("Melatih model Linear Regression...")
lr_model = LinearRegression()  # Inisialisasi model
lr_model.fit(X_train_scaled, y_train)  # Melatih model dengan data training

# Melatih model Random Forest
print("Melatih model Random Forest...")
# n_estimators=100 berarti menggunakan 100 pohon keputusan
# random_state=42 untuk hasil yang konsisten
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)  # Melatih model dengan data training

# Menyimpan model-model yang telah dilatih
print("Menyimpan model dan scaler...")
# Menyimpan model Linear Regression
with open('boston_house_lr_model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)
print("Model Linear Regression telah disimpan sebagai 'boston_house_lr_model.pkl'")

# Menyimpan model Random Forest
with open('boston_house_rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
print("Model Random Forest telah disimpan sebagai 'boston_house_rf_model.pkl'")

# Menyimpan scaler untuk preprocessing data baru nantinya
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler telah disimpan sebagai 'scaler.pkl'")

print("Proses training dan penyimpanan selesai!")

# Evaluasi performa model
print("\nEvaluasi Model:")
# Evaluasi Linear Regression
lr_pred = lr_model.predict(X_test_scaled)  # Prediksi menggunakan data test
lr_score = lr_model.score(X_test_scaled, y_test)  # Menghitung R2 score
print(f"Skor R2 Linear Regression: {lr_score:.4f}")

# Evaluasi Random Forest
rf_pred = rf_model.predict(X_test_scaled)  # Prediksi menggunakan data test
rf_score = rf_model.score(X_test_scaled, y_test)  # Menghitung R2 score
print(f"Skor R2 Random Forest: {rf_score:.4f}")

# Menambahkan analisis performa model
print("\nAnalisis Tambahan:")
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Menghitung error metrics untuk Linear Regression
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_pred)
print("\nMetrik Error Linear Regression:")
print(f"Mean Squared Error (MSE): {lr_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {lr_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {lr_mae:.2f}")

# Menghitung error metrics untuk Random Forest
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
print("\nMetrik Error Random Forest:")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")