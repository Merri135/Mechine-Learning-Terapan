# Mechine Learning Terapan_Merri Putri.ipynb
# original file located at : https://colab.research.google.com/drive/1lDQUkxB10xgSiUK8zoztDG7W7VNc_QMg#scrollTo=RpJimTepuS1f
# Nama : Merri Putri Panggabean
# ID Chorot : mc404d5x0047
#  Dataset yang digunakan : https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

# Import Library yang digunakan.
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import  OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor  # Import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Pipeline
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

# Menginstall Package Kaggle
# !pip install -q kaggle
# mengupload Kaggle.Json
# from google.colab import files
# files.upload()
# # membuat direktori dan mengubah izin
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !ls ~/.kaggle
# !kaggle datasets download -d spscientist/students-performance-in-exams

# # Ekstrak ZIP
from zipfile import ZipFile
file_name ="students-performance-in-exams.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("Selesai Di ekstrak")


# load dataset
insu_df = pd.read_csv('StudentsPerformance.csv')
insu_df.head(10)

# Exploratory Data Analysis (EDA)

# Melihat Informasi dari Dataset
print(f'Jumlah Baris: {insu_df.shape[0]} | Jumlah Kolom: {insu_df.shape[1]}')
insu_df.info()

# Melihat statistik dari dataset
insu_df.describe()

# Melihat Missing Value dan data Duplikat
print("Jumlah Missing Value per Kolom:")
print(insu_df.isnull().sum())
# melihat duplikat data
num_duplicates = insu_df.duplicated().sum()
print(f"Jumlah duplikat: {num_duplicates}")

# Exploratory Data / Visualisasi Data

# Cek semua kolom numerik dengan boxplot
# untuk Menampilkan Outlier pada Data Numerik.
numerical_cols = ['math score', 'writing score', 'reading score']

plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=insu_df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()
# Menangani Outlier
numeric_df = insu_df.select_dtypes(include=['number']) # Select only numerical columns
Q1 = numeric_df.quantile(0.25) # Calculate quantiles on the numerical DataFrame
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
insu_df = insu_df[~((insu_df[numeric_df.columns] < (Q1 - 1.5 * IQR)) | (insu_df[numeric_df.columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
# Cek ukuran dataset setelah kita drop outliers
insu_df.shape
# Menampilkan hasil setelah Outlier.
# Pilih kolom numerik
numerical_cols = ['math score', 'writing score', 'reading score']

# Buat boxplot untuk masing-masing kolom numerik
plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=insu_df[col], color='skyblue')
    plt.title(f"Boxplot of {col} (Setelah Outlier Dihapus)")
    plt.tight_layout()
plt.show()

# Exploratory Data Analysis - Univariate Analysis

# Membagi menjadi 2 data kolom yaitu Numerik dan Kategori.
num_features = ['math score', 'writing score', 'reading score']
cat_features = ['parental level of education','race/ethnicity','test preparation course','lunch','gender']
#  Distribusi kolom numerik
# Loop untuk membuat plot distribusi tiap kolom numerik
for col in num_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(insu_df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Pastikan semua label kategori terlihat
for col in cat_features:
    plt.figure(figsize=(8, 4))  # bisa disesuaikan sesuai panjang label
    sns.countplot(data=insu_df, x=col, order=insu_df[col].value_counts().index)
    plt.title(f'Distribusi {col}')
    plt.ylabel('Count')
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')  # putar label biar muat & rapi
    plt.tight_layout()
    plt.show()
# Loop untuk setiap fitur kategorikal
for feature in cat_features:
    print(f"\n==== Ringkasan untuk fitur: {feature} ====")

    count = insu_df[feature].value_counts()
    percent = 100 * insu_df[feature].value_counts(normalize=True)

    df1 = pd.DataFrame({
        'jumlah sampel': count,
        'persentase (%)': percent.round(1)
    })

    print(df1)

    # Visualisasi bar chart
    count.plot(kind='bar', title=feature, figsize=(6, 3), color='skyblue')
    plt.ylabel('Jumlah')
    plt.xlabel(feature)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
insu_df.hist(bins=50, figsize=(20,15))
plt.show()
# Boxplot kolom math score terhadap fitur kategori
for col in cat_features:
    plt.figure(figsize=(8, 6))  # Ukuran figure yang lebih proporsional
    sns.barplot(data=insu_df, x=col, y='math score', palette='Set2')
    plt.title(f'Rata-rata nilai matematika (math score) berdasarkan {col}')
    plt.xlabel(col)
    plt.ylabel('Rata-rata Nilai Matematika')  # Perbaikan label sumbu y
    plt.xticks(rotation=45, ha='right')  # Rotasi label sumbu x agar lebih mudah dibaca
    plt.tight_layout()
    plt.show()
# Barplot untuk distribusi Nilai Writing berdasarkan Pendidikan Orang Tua ---
plt.figure(figsize=(12, 6))
sns.barplot(data=insu_df, x='parental level of education', y='writing score', hue='parental level of education', palette='Set2', errorbar=None)
plt.title('Perbandingan Nilai Writing Berdasarkan Pendidikan Orang Tua')
plt.xticks(rotation=45)  # Rotasi label sumbu x agar lebih terbaca
plt.legend(title='Pendidikan Orang Tua', loc='upper left', bbox_to_anchor=(1, 1))  # Menambahkan legenda untuk hue
plt.show()

# Barplot untuk distribusi Nilai Writing berdasarkan Test Preparation Course ---
plt.figure(figsize=(12, 6))
sns.barplot(data=insu_df, x='test preparation course', y='writing score', hue='gender', palette='Set1', errorbar=None)
plt.title('Perbandingan Nilai Writing Berdasarkan Kursus Persiapan dan Gender')
plt.legend(title='Gender', loc='upper left', bbox_to_anchor=(1, 1))  # Menambahkan legenda untuk hue
plt.show()
# Barplot untuk distribusi Nilai Reading berdasarkan Pendidikan Orang Tua ---
plt.figure(figsize=(12, 6))
sns.barplot(data=insu_df, x='parental level of education', y='reading score', hue='parental level of education', palette='Set2', errorbar=None)
plt.title('Perbandingan Nilai Reading Berdasarkan Pendidikan Orang Tua')
plt.xticks(rotation=45)  # Rotasi label sumbu x agar lebih terbaca
plt.legend(title='Pendidikan Orang Tua', loc='upper left', bbox_to_anchor=(1, 1))  # Menambahkan legenda untuk hue
plt.show()

#  Barplot untuk distribusi Nilai Reading berdasarkan Test Preparation Course ---
plt.figure(figsize=(12, 6))
sns.barplot(data=insu_df, x='test preparation course', y='reading score', hue='gender', palette='Set1', errorbar=None)
plt.title('Perbandingan Nilai Reading Berdasarkan Kursus Persiapan dan Gender')
plt.legend(title='Gender', loc='upper left', bbox_to_anchor=(1, 1))  # Menambahkan legenda untuk hue
plt.show()
# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(insu_df, diag_kind = 'kde')
# Korelasi antar fitur numerik
plt.figure(figsize=(8, 6))
sns.heatmap(insu_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi antar Fitur Numerik')
plt.tight_layout()
plt.show()

# Data Prepration 

# Melakukan Standarisasi dan menimpa kolom asli
df = insu_df.copy()

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

print(df)
# One-Hot Encoding per kolom, gabung ke DataFrame
df = pd.concat([df, pd.get_dummies(df['gender'], prefix='gender')], axis=1)
df = pd.concat([df, pd.get_dummies(df['lunch'], prefix='lunch')], axis=1)
df = pd.concat([df, pd.get_dummies(df['parental level of education'], prefix='parental level of education')], axis=1)
df = pd.concat([df, pd.get_dummies(df['test preparation course'], prefix='test preparation course')], axis=1)
df = pd.concat([df, pd.get_dummies(df['race/ethnicity'], prefix='race/ethnicity')], axis=1)
# Drop kolom aslinya (karena sudah di-encode)
df.drop(['test preparation course','race/ethnicity','parental level of education','gender','lunch'], axis=1, inplace=True)

# Tampilkan hasil
df.head()
sns.pairplot(df[['math score', 'gender_male', 'gender_female']],
             hue='math score', plot_kws={"s": 80})
plt.show()
pca = PCA(n_components=3, random_state=123)
pca.fit(df[['math score', 'writing score', 'reading score']])
princ_comp = pca.transform(df[['math score', 'writing score', 'reading score']])
print(pca)
pca.explained_variance_ratio_.round(3)
print(df)
# Fitur dan Target
X = df.drop(["math score"],axis =1)
y = df["math score"]
# Split Data
# melakukan pembagian data X dan y dengan train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(f'Total jumlah sample pada dataset: {len(X)}')
print(f'Total jumlah sample pada train dataset: {len(X_train)}')
print(f'Total jumlah sample pada test dataset: {len(X_test)}')

# Model Development
# Siapkan dataframe untuk analisis model
df_models = pd.DataFrame(index=['Train MSE', 'Test MSE'],
                      columns=['KNN', 'RandomForest', 'Boosting','SVM'])

# Model 1 K-Nearest Neighborn
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
df_models.loc['Train MSE','KNN'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

# Prediksi di training set
y_train_pred = knn.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Prediksi di test set
y_test_pred = knn.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Model 2 Random Forest
# buat model prediksi
rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
rf.fit(X_train, y_train)

df_models.loc['Train MSE','RandomForest'] = mean_squared_error(y_pred=rf.predict(X_train), y_true=y_train)

# Prediksi di training set
y_train_pred = rf.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Prediksi di test set
y_test_pred = rf.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Model 4 Boosting
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
df_models.loc['Train MSE','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

# Prediksi di training set
y_train_pred = boosting.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Prediksi di test set
y_test_pred = boosting.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Model 5 Support Vector Mechine (SVM)
# Melatih Model
svm= SVR()
svm.fit(X_train, y_train)
df_models.loc['Train MSE','SVM'] = mean_squared_error(y_pred=svm.predict(X_train), y_true=y_train)

# Prediksi di training set
y_train_pred = svm.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Prediksi di test set
y_test_pred = svm.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['Train', 'Test'], index=['KNN','RF','Boosting','SVM'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': rf, 'Boosting': boosting,'SVM':svm}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'Train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'Test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))
 # Panggil mse
mse

mse

# Melihat MSE pada setiap model yang dilatih.
fig, ax = plt.subplots()
mse.sort_values(by='Test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# Contoh: prediksi dari model-model
pred_1 = knn.predict(X_test)         # KNN
pred_2 = rf.predict(X_test)          # Random Forest
pred_3 = boosting.predict(X_test)
pred_4 = svm.predict(X_test)

# Gabungkan semua hasil ke dalam DataFrame
df_prediksi = pd.DataFrame({
    'y_true': y_test,
    'prediksi_KNN': pred_1,
    'prediksi_RF': pred_2,
    'prediksi_Boosting': pred_3,
    'prediksi_svm': pred_4
})

# Tambahkan index asli untuk pelacakan (optional)
df_prediksi.index.name = 'index_sample'

# Tampilkan beberapa baris untuk dicek
print(df_prediksi.sample(2).sort_index())

