# Laporan Proyek Machine Learning - Merri Putri Panggabean

## Domain Proyek

###  Latar Belakang 

Pendidikan merupakan salah satu pilar utama dalam pembangunan suatu bangsa. Kualitas pendidikan tidak hanya ditentukan oleh kurikulum dan fasilitas, tetapi juga oleh berbagai faktor individu dan sosial yang memengaruhi hasil belajar siswa. Salah satu cara untuk mengevaluasi hasil belajar siswa adalah melalui performa mereka dalam ujian akademik.
  
## Business Understanding

Dalam dunia pendidikan, memahami faktor-faktor yang memengaruhi performa siswa sangat krusial untuk meningkatkan kualitas pembelajaran dan mencegah ketimpangan hasil akademik. Melalui pendekatan analisis data dan machine learning, kita dapat mengklarifikasi permasalahan, merumuskan tujuan, serta menyusun solusi prediktif untuk membantu pengambilan keputusan di bidang pendidikan.

### Problem Statements
Berdasarkan latar belakang diatas, berikut batasan masalah yang akan diselesaikan dalam proyek ini :
- Apa saja faktor yang berpengaruh terhadap nilai ujian siswa?
- Bagaimana perbandingan persentase siswa yang lulus dan gagal berdasarkan akademik
- Bagaimana kita bisa memprediksi performa siswa pada nilai skor ujian membaca menggunakan KNN,SVM,RF dan BOOSTING?

### Goals

Menjawab batasan masalah yang telah dirangkum dan akan menjawab pertanyaan-pertanyaan diatas, berikut tujuan masalahnya :
- Kita perlu untuk menganalisis faktor yang berpengaruh pada nilai skor siswa.
- Kita perlu untuk melakukan visualisasi untuk melihat persentase siswa yang lulus dan gagal berdasarkan akademik
- Kita perlu untuk membangun model prediksi performa siswa menggunakan ke empat algoritma yaitu KNN,RF,SVM dan BOOSTING.

### Solution Statment

untuk mencapai tujuan diatas, maka kita perlu melakukan pendekatan pemodelan yang dimana kita akan menggunakan langkah-langkah seperti ini :
- Analisis  Deskriptif : kita akan menggunakan statistik deskriptif untuk memahami distribusi data. ini mencakup semua nilai skor akademik.
- Analisis korelasi : menggunakan korelasi untuk melihat hubungan antara berbagai faktor yang mempengaruhi skor nilai siswa.
- Model Prediksi : membangun model predeksi berbasis mechine learning seperti Random Forest (RF), K-Nearest Neighbors (KNN), Boosting, Support Vector Machine (SVM) untuk menangani nilai skor ujian siswa.

**Menggunakan MSE (Mean Squared Error) pada model yang akan di evaluasi.**

## Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi tentang performa akademik siswa berdasarkan beberapa faktor demografis dan sosial, seperti jenis kelamin, kelompok etnis, tingkat pendidikan orang tua, status mengikuti kursus persiapan ujian, dan jenis makan siang yang dikonsumsi. Selain itu, dataset ini juga mencatat skor ujian siswa dalam tiga mata pelajaran utama, yaitu **matematika (math score), membaca (reading score), dan menulis (writing score).**

berikut link dataset yang dianalisa : https://www.kaggle.com/datasets/spscientist/students-performance-in-exams 

Dataset mentah yang digunakan dalam proyek ini memiliki 1000 baris data dan 8 kolom. Kolom - kolom tersebut terdiri dari 5 kolom kategori dan 3 kolom numerik. untuk penjelasan mengenai variabel-variabel pada Performance Students dataset adalah sebagai berikut:
- gender : jenis kelamin tiap siswa.
- lunch : jenis makan siang yang setiap hari di konsumsi tiap siswa.
- race/ethnicity : jenis kelompok etnis tiap siswa. yang sering dikategorikan dari group A sampai group E.
- parental level of education : tingkat pendidikan orang tua (misalnya: high school, bachelor’s degree).
- test preparation course : status mengikuti kursus persiapan ujian.
- math score : nilai skor ujian matematika.
- reading score : nilai skor ujian membaca.
- writing score : nilai skor ujian menulis.

### Langkah - Langkah Pemrosesan data
- memasukkan dataset kedalam dataframe menggunakan library pandas.<br>
- menampilkan informasi dari dataset.<br>
- menampilkan jumlah data statistik pada dataset.<br>
- menampilkan jumlah missing value dan duplikat data pada dataset.<br>
- menangani outlier.<br>
- menampilkan visualisasi antar fitur numerik antar kategorial.<br>

#### Membuat dataframe dengan library pandas.
Pada proyek ini, menggunakan fungsi **.read** untuk memasukkan dataset PerformanceStudents.csv kedalam bentuk dataframe menggunakan library pandas dan dataframe yang tersimpan ialah **insu_df**. setelah itu, untuk menampilkan 10 data pertama maka menggunakan **.head(10)**. maka berikut tampilannya :

![alt text](./asset/head.png)

Gambar 1. tampilan 10 data pertama pada dataset insu_df.

#### Menampilkan informasi dari dataset
Pada proyek ini,untuk memahami semua atribut-atribut yang dipakai, memakai fungsi library python yaitu .info() untuk menampilkan atribut-atribut dan baris serta kolom pada dataset. dimana perintah .info() berfungsi untuk menampilkan semua tipe data pada masing-masing atribut dalam dataset.

![alt text](./asset/info.png)

gambar 2. tampilan informasi pada dataset insu_df.

dari output pada gambar diatas, maka dilihat bahwa :
1. terdapat 3 kolom numerik yaitu **math score, writing score dan reading score**.
2. terdapat 5 kolom kategori yaitu **test preparation course, race/ethnicity, parental level of education, lunch dan age**.
3. memiliki jumlah 1000 baris dan 8 kolom pada dataset.
   
#### Menampilkan data statistik dataset.
Pada proyek ini, menggunakan perintah .describe() untuk menampilkan dan mengetahui statistik dasar dari kolom **math score, reading score, writing score** seperti percentile, mean, standar deviasi, jumlah data, min dan max. maka berikut tampilannya :

![alt text](./asset/describe.png)

gambar 3. keluaran statistik pada dataset insu_df menggunakan fungsi .describe(). 

#### Menampilkan missing value dan duplikat data pada dataset.
Pada proyek ini, untuk mengetahui data memiliki missing value dan duplikat data, maka menggunakan perintah fungsi .insnull() dan .duplicated() pada dataset insu_df, setelah melakukan perintah tersebut, maka akan menampilkan kolom yang memiliki missing value dan jumlah duplikat. maka beikut tampilannya :

![alt text](./asset/missing.png)<br>
Gambar 4. tampilan missing value.

![alt text](./asset/duplikat.png)<br>
Gambar 5. tampilan duplikat data.<br>
Pada gambar diatas menunjukkan bahwa dataset PerformanceStudents bersih tanpa missing value dan duplikat data.

## Exploratory Data Analysis
### Melihat outlier pada dataset.
Pada kasus ini, kita akan melihat outlier dalam setiap kolom pada dataset, outlier sendiri adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari hasil pengamatan lainnya. maka berikut tampilan kolom yang outlier :
![alt text](./asset/sebelum.png)<br>
Gambar 6. tampilan sebelum menangani outlier.<br>
dari gambar diatas, terdapat outlier pada kolom math score,reading score dan writing score. untuk menangani outlier kita dapat menggunakan teknik IQR method. IQR adalah *Interquartile Range*. berikut rumus akan kita pakai :<br>
![alt text](./asset/outlier.png)<br>
setelah melakukan penanganan outlier pada kolom dataset, kita dapat melihat hasil dari penanganan outlier yang telah kita lakukan, maka tampilan hasil penganganan outlier, sebagai berikut :<br>
![alt text](./asset/setelah.png)<br>
Gambar 7. setelah melakukan pengangan outlier.<br>

#### Menampilkan distribusi numerik dan kategori

**Visualisasi Numeric Fitur**
- Numeric math score<br>
  ![alt text](./asset/Figure3.png)<br>
  Gambar 8. hasil visualisasi chart bar **math score**.<br>
  Pada gambar 8 merupakan hasil visualisasi kolom **math score** yang dapat kita lihat bahwa jumlah nilai skor matematika siswa tersebar cukup merata dengan kecenderungan siswa memiliki nilai antara 60-70, distribusi tidak sepenuhnya simetris, dikarenakan ada leih sedikit siswa mendapat nilai tertinggi dan sangat rendah.
- Numeric writing score<br>
![alt text](./asset/Figure4.png)<br>
Gambar 9. Hasil Visualisasi chart bar **writing score**.<br>
Pada gambar 9 menampilkan visualisasi writing score yang dapat kita lihat bahwasebagian besar peserta mendapatkan skor menulis di sekitar nilai 60-80, distribusi nilai skor cenderung simetris sekitar nilai 70 dan mengindikasi bahwa skor-skor secara cukup merata disekitar rata-rata.
- Numeric reading score<br>
![alt text](./asset/Figure5.png)<br>
Gambar 10. Hasil visualisasi chart bar **reading score**<br>
  Pada gambar 10 menampilkan hasil visualisasi rading score yang dapat kita lihat bahwa sebagian siswa mendapatkan nilai skor antara 60-80 serta nilai yang sangat rendah antar 40 dan tertinggi antara 90 tetapi jarang diperoleh.

**Visualisasi Kategori Fitur**
- Kategori gender<br>
  ![alt text](./asset/Figure_12.png)<br>
  Gambar 11. Hasil Visualisasi kolom **gender**.<br>
  Pada gambar 11 merupakan hasil dari visualisasi yang dilakukan, dapat kita lihat bahwa jenis kelamin female lebih tinggi dari jenis kelamin male.
- Kategori lunch<br>
  ![alt text](./asset/Figure_11.png)<br>
  Gambar 12. Hasil visualisasi kolom **lunch**.<br>
  Pada gambar 12, dapat kita lihat bahwa jenis makan siang yang lebih tinggi ialah *standar* daripada *free/recuded* yang lebih rendah.
- Kategori Parental level of education
  ![alt text](./asset/Figure_7.png)<br>
Gambar 13. Hasil visualisasi kolom **parental level of education**.<br>
Dari gambar 13 merupakan hasil dari visualisasi yang kita lakukan, maka dapat kita lihat bahwa tingkat pendidikan *some college* lebih tinggi dari data lainnya sedangkan yang terendah ialah tingkat pendidikan *master degree*.
-  Kategori race/ethnicity<br>
![alt text](./asset/Figure_8.png)<br>
Gambar 14. Hasil visualisasi kolom **race/ethnicity**.<br>
Pada gambar 14, dapat kita lihat bahwa kelompok etnis yang lebih tinggi ialah *group c* sedangkan *group B* cenderung signifikan, dan kelompok etnis yang paling rendah ialah *group A*.
- Kategori test preparation course<br>
![alt text](./asset/Figure_9.png)<r>
Gambar 15. Hasil visualisasi kolom **test preparation course**<br>
Pada gambar 15 merupakan hasil visualisasi yang kita lakukan, maka dapat kita lihat bahwa jenis ujian yang paling tinggi ialah *none* dibandingkan jenis ujian *completed* memiliki nilai yang paling rendah.

**Visualisasi History kolom numerik<br>**
![alt text](./asset/history.png)<br>
Gambar 16. History kolom numerik<br>
Dari gambar diatas, kita dapat lihat bahwa ketiga mata pelajaran memiliki distribusi mendekati normal, dapat kita lihat dari mata pelajaran *matematika* sedikit lebih *rendah* diantara 60-70, mata pelajaran *reading* cenderung lebih sedikit lebih tinggi diantara 65-75 dan mata pelajaran *writing* hampir sama dengan mata pelajaran *matematika* hanya 0.5% naik dibanding *matematika*.

**Visualisasi korelasi data numerik terhadap data kategori<br>**

![alt text](./asset/parental.png)<br>
![alt text](./asset/race.png)<br>
![alt text](./asset/test.png)<br>
![alt text](./asset/lunch.png)<br>
![alt text](./asset/gender.png)<br>
Gambar 17. Visualisasi korelasi data numerik pada data kategori.<br>
Pada gambar 17 merupakan hasil visualisasi korelasi semua *mata pelajaran* pada data kategori. dimana yang kita lihat terdapat bar chart relatif sama rata, tetapi ada juga perbedaan yang signifikan yaitu :

- Pada bar chart *parental level of education* yang kita lihat bahwa semakin tinggi tingkat pendidikan orang tua maka skor nilai akademik tiap siswa lebih tinggi dan semakin rendah tingkat pendidikan orangtua maka skor nilai akademik tiap siswa lebih rendah. maka jika kita menganalisa dari gambar diatas bahwa rentang nilai antar **tingkat pendidikan orang tua terlihat lebih lebar pada mata pelajaran membaca dan menulis** dibandingkan dengan **matematika.** Ini mengindikasikan bahwa tingkat pendidikan orang tua mungkin memiliki pengaruh yang lebih besar pada kemampuan membaca dan menulis siswa.
- dari bar chart *race/ethinicity* yang kita lihat bahwa Kelompok E cenderung memiliki rata-rata nilai tertinggi secara keseluruhan dibandingkan kelompok lain, **terutama dalam mata pelajaran membaca dan menulis** dan Kelompok A cenderung memiliki rata-rata nilai terendah di antara kelompok lain dalam ketiga mata pelajaran. 
- dari bar chart *test preparation course* yang kita lihat bahwa ada perbedaan yang relatif tidak merata, dimana siswa yang mengikuti kursus persiapan tes cenderung memiliki rata-rata nilai yang lebih tinggi secara signifikan dalam ketiga mata pelajaran *matematika, membaca, dan menulis* dibandingkan dengan siswa yang tidak mengikuti kursus. tetapi dari yang kita ketahui bahwa perbedaan rata-rata nilai antara kedua kelompok **mengikuti kursus vs. tidak mengikuti kursus** tampak **paling besar pada mata pelajaran membaca dan menulis** dibanding dengan mata pelajaran **matematika**.
- dari bar chart *lunch* yang kita ketahui bahwa siswa yang mendapatkan **makan siang standar cenderung memiliki rata-rata nilai yang lebih tinggi** dalam ketiga mata pelajaran *matematika, membaca, dan menulis* dibandingkan dengan siswa yang mendapatkan **makan siang gratis**. walaupun begitu, perbedaan rata-rata nilai tampak paling besar pada **mata pelajaran membaca** dibanding dengan **mata pelajaran matematika dan menulis.**
- dari bar chart *gender* yang kita ketahui bahwa terdapat perbedaan rata-rata nilai antara siswa perempuan dan laki-laki dalam mata pelajaran membaca dan menulis. **Siswa perempuan** cenderung memiliki rata-rata skor yang **lebih tinggi** dalam kedua mata pelajaran ini dibandingkan **siswa laki-laki** dan kita ketahui bahwa **siswa perempuan** menunjukkan keunggulan yang lebih jelas dalam kemampuan verbal **membaca dan menulis**, sementara performa dalam **matematika** hampir setara dengan **siswa laki-laki.**


**Korelasi matriks fitur numerik.<br>**
![alt text](./asset/matrik.png)<br>
Gambar 19. Korelasi matrik fitur numerik.<br>
Pada gambar diatas merupakan hasil korelasi matrik pada fitur numerik, yang dimana diketahui bahwa setiap dalam sel adalah nilai koefisien korelasi pearson antara dua fitur. dimana nilai antara 1 dan -1 menunjukkan korelasi yang kuat sedangkan nilai yang mendekati 0 menunjukkan korelasi yang lemah.<br>
dari hasil visualisasi yang kita ketahui bahwa fitur 'math score' dan 'writing score' keduanya memiliki hubungan yang positif dengan 'reading score'. jadi, fitur 'reading score' berkorelasi tinggi dengan kedua fitur tersebut.
## Data Preparation
### Tahap Preparation :**
- mengubah data kategori pada dataset menjadi 'true' dan 'false' dengan menggunakan One-Hot-Encoding.<br>
- melakukan data splitting menjadi data latih dan data test.<br>
- melakukan fungsi 'Standarisasi' pada data numerik<br>

**One-Hot -Encoding pada data numerik<br>**

| index | math score | reading score | writing score | gender_female | gender_male | lunch_free_reduced | lunch_free_standard | parental level of education_associate's degree | parental level of education_bachelor's degree | parental level of education_high school | parental level of education_master's degree | parental level of education_some college | parental level of education_some high school | test preparation course_completed | test preparation course_none | race/ethnicity_groupA | race/ethnicity_groupB | race/ethnicity_groupC | race/ethnicity_groupD | race/ethnicity_groupE |
|:-----:|:----------:|:-------------:|:-------------:|:-------------:|:-----------:|:------------------:|:-------------------:|:----------------------------------------------:|:--------------------------------------------:|:--------------------------------------:|:-------------------------------------------:|:--------------------------------:|:----------------------------------:|:-----------------------------:|:-----------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| 0     | 0.373174   | 0.168406       | 0.374241       | True          | False       | False              | True                | False                                         | True                                       | False                               | False                                    | False                         | False                           | True                        | False                 | True                | False               | False               | False               | False               |
| 1     | 0.164871   | 1.453233       | 1.338567       | True          | False       | False              | True                | False                                         | False                                      | False                               | False                                    | True                          | False                           | True                        | False                 | False               | False               | True                | False               | False               |
| 2     | 1.622992   | 1.810130       | 1.682969       | True          | False       | False              | True                | False                                         | False                                      | False                               | True                                     | False                         | False                           | False                       | True                  | False               | True                | False               | False               | False               |
| 3     | -1.362684  | -0.902283      | -1.692172      | False         | True        | True               | False               | True                                          | False                                      | False                               | False                                    | False                         | False                           | False                       | True                  | True                | False               | False               | False               | False               |
| 4     | 0.650912   | 0.596682       | 0.443121       | False         | True        | False              | True                | False                                         | False                                      | False                               | False                                    | True                          | False                           | False                       | True                  | False               | False               | True                | False               | False               |
<br>
Tabel 2. One-Hot-Encoding pada data kategori.<br>
Pada tabel diatas merupakan hasil one hot encoding yang kita lakukan pada data kategori yang menghasilkan *False* dan *True* pada semua data bertipe kategori. menggunakan one-hot-encoding dengan teknik **.get_dummies()**

**Data Splitting<br>**
Pada proyek ini kita akan menggunakan data split untuk membagi fitur target yang akan kita latih selanjutnya. untuk melakukan itu, perlu mengimport library data split yaitu *train_test_split*, kemudian membagi variabel menjadi 2 buah yaitu X yang berfungsi untuk menghapus kolom *reading score* dan y untuk menampilkan kolom *reading score* lalu dibagi menjadi 4 variabel baru yaitu *X_train, X_test,y_train,y_test menggunakan library *train_test_split* dengan parameter seperti ini :
- X berfungsi untuk menghapus kolom *reading score*
- y berfungsi untuk menampilkan target yaitu kolom *reading score*
- test_size adalah ukuran pembagian dataset yang akan kita lakukan, dengan ketentuan 80% untuk data training dan 20% data testing.
- random_state digunakan untuk mengontrol random numer generator yang akan digunakan, maka penulis menggunakan **random_state=42**.

setelah melakukan pembagian data pada data splitting, kita bisa mengetahui berapa banyak jumlah sampel pada setiap data yang telah kita bagikan sebelumnya, untuk menampilkan jumlah sampel pada setiap data yang dibagi ialah menggunakan fungsi *len(X_train) dan len(X_test)*. maka berikut hasil split dataset.

![alt text](./asset/sampel.png)<br>
Gambar 20. hasil jumlah sample<br>

**Standarisasi<br>**
Proses Scaling dan Standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. standarisasi adalah teknik tansformasi yang digunakan dalam tahap persiapan pemodelan dengan menggunakan teknik *StandarScaler* dari library *Scikitlearn*.

StandarScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) setelah itu membaginya dengan standar deviasi, standarsclaer menghasilkan distribusi dengan standar deviasi ialah 1 dan mean ialah 0. ini berfungsi untuk menghindari kebocoran informasi pda data uji.
## Modeling
Penulis menerapakan 4 algoritma model mechine learning yang berbeda ialah :
1. K-Nearest Neighbors (KNN)<br>
2. Random Forest<br>
3. ADABOOOST<br>
4. Support Vector Mechine (SVM)<br>

#### K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).<br>
**Cara Kerja KNN:<br>**

- menentukan nilai K (jumlah tetangga terdekat).
- menghitung jarak dengan menggunakan rumus *Euclidean* antara data yang ingin diprediksi.
- mengambil nilai K terdekat.
- menampilkan nilai K terbaik.<br>
**Kelebihan dan kekurangan KNN<br>**
  
- **Kelebihan** pada KNN ialah mudah diimplementasikan, cocok untuk data kecil serta tidak memerlukan pelatihan model lainnya.
- meskipun KNN memiliki kelebihan, maka KNN juga memiliki **Kekurangan** ialah sensitif terhadap fitur yang tidak sesuai, kurang efektif jika terdapat data yang noise dan lambat untuk dataset besar.<br>
#### Random Forest
Random Forest adalah algoritma ensemble learning.ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir.<br>

**Cara Kerja Random Forest<br>**
- membuat keputusan dari subset acak data
- menggabungkan prediksi dari semua pohon<br>
pada kasus proyek ini bertipe regresi maka digunakan random forest Regressor dari library Scikit-learn dengan beberapa parameter yang digunakan:
- n_estimator : jumlah trees (pohon) di forest. disini kita menerapkan **n_estimator=50**.
- max_depth ialah kedalaman atau panjang pohon.bertujuan untuk membagi setiap node ke dalam jumlah pengamatan yang dihasilkan. penulis menerapkan **max_deptg=16**.
- random_state digunakan untuk mengontrol random number generator yang digunakan, penulis menerapkan **random_state=55**
- n_jobs ialah jumlah job yang digunakan secara paralel.penulis menerapkan **n_jobs=-1** artinya semua proses berjalan secara paralel.<br>

**Kelebihan dan kekurangan Random Forest<br>**
- **Kelebihan** pada Random Forest ialah menghasilkan akurasi tinggi, tidak mudah overfitting dan dapat menangani data besar.
- **Kekurangan** ialah hasil kurang interpreatif dan ukuran model besar sulit untuk dikembangkan.<br>

#### ADABOOST / BOOSTING
Boosting merupakan teknik ensemble yang menggabungkan beberapa model lemah (weak learners) secara berurutan untuk membuat model kuat (strong learner).<br>
**Cara Kerja Boosting<br>**
- melatih model lemah pertama.
- menghitung eror serta membuat model berikutnya untuk memperbaiki eror.
- menggabungkan semua model dengan bobot tertentu serta mengulangi sampah model cukup baik mencapai batas iterasi.<br>

**Kelebihan dan kekurangan Boosting <br>**
- **Kelebihan** pada Boosting ialah dapat mengangani hubungan non-linear yang kompleks dan mudah untuk di implementasikan.
- **Kekurangan** pada Boosting ialah rentan overfitting jika tidak dikontrol dan banyak parameter yang harus dituning.<br>

#### Support Vector Mechine (SVM)
Support Vector Mechine (SVM) merupakan untuk menemukan hyperplane terbaik untuk memisahkan data dari dua kelas dengan margin maksimun.<br>

**Kekurangan dan kelebihan SVM<br>**
- **Kelebihan** pada SVM ialah akurat untuk data dengan margin yang jelas , efektif di ruang dimensi tinggi dan mendukung kernel untuk data non-linear.
- **Kekurangannya** ialah tidak cocok untuk dataset besar sarta perlu penyesuaian parameter.<br>
Pada proyek ini, penulis menggunakan **SVR (Support Vector Regresi)**. Cara kerja SVR ialah :
- berusaha meminimalkan eror semua data, tapi hanya meminimalkan error yang berada di luar margin epsilon.
- membangun sebuah garis yang memiliki deviasi paling keci terhadap semua data.

## Evaluation
Pada proyek ini menggunakan model mechine learning bertipe **Regresi**. Metrik yang digunakan untuk melakukan Evaluasi model ialah MSE (Mean Squared Error) yang dimana bertujuan untuk mengukur rata-rata kuadrat selisih antara prediksi dan nilai aktual. berikut rumus MSE :<br>
![alt text](./asset/mse.png)<br>
Gambar 21. Rumus MSE<br>
keterangan :<br>
n = jumlah dataset<br>
yi = nilai sebenarnya<br>
y_pred = nilai yang diprediksi.<br>

|           | Train      | Test       |
|-----------|------------|------------|
| **KNN**   | 23.643722  | 27.660707  |
| **RF**    | 2.802247   | 19.136243  |
| **Boosting** | 19.132377  | 20.327075  |
| **SVM**   | 25.240321  | 29.443402  |
<br>
Tabel 3. Hasil MSE.<br>
Pada tabel diatas merupakan hasil MSE yang telah kita lakukan pada data train dan data test. untuk lebih memudahkan penulis menampilkan plot matrik dengan bar chart.

![alt text](./asset/model.png)<br>

Gambar 22. Visualisasi hasil MSE dari ke 4 algoritma<br>
Dari gambar diatas, terlihat bahwa model Random forest pada data train memiliki nilai error yang sangat kecil tetapi pada data test memiliki nilai yang tinggi yang mengalami data test overfitting. dibanding dengan model KNN dan SVM memiliki nilai error pada data train dan data test yang tinggi dan untuk model boosting relatif seimbang pada data train dan data test. sehingga model Random Forest yang akan kita pilih sebagai model terbaik untuk melakukan prediksi hasil nilai skor ujian membaca pada siswa.

untuk mengujinya, penulis membuat prediksi menggunakan beberapa harga dari data test.

![alt text](./asset/harga.png)<br>
Gambar 23. Hasil prediksi MSE<br>
Pada gambar diatas adalah hasil prediksi *'reading score'* dari ke empat algoritma yaitu KNN, Random Forest (RF), Boosting, dan SVM terhadap dua sampel data. Nilai asli (y_true) dibandingkan dengan hasil prediksi dari masing-masing model.

Terlihat bahwa pada sampel pertama (index 68), keempat model memberikan prediksi yang sangat dekat dengan nilai aktual (58), dan Boosting memberikan hasil yang paling mendekati (58.2018). Sedangkan pada sampel kedua (index 214), keempat model cenderung underpredict, dan SVM memberikan hasil prediksi terdekat dengan nilai sebenarnya (91).

Berdasarkan hasil prediksi pada dua sampel, terlihat bahwa model Random Forest (RF) menghasilkan nilai prediksi MSE terkecil sebesar 444.87, diikuti oleh SVM dengan nilai MSE sebesar 486.92, lalu KNN sebesar 616.01, dan Boosting menghasilkan MSE tertinggi sebesar 761.55.

Hal ini menunjukkan bahwa model RF lebih akurat dibandingkan SVM, Boosting maupun KNN dalam memprediksi nilai target.

## Kesimpulan 
Dapat dilihat dari empat model algoritma yang diuji, yaitu KNN, Random Forest, Boosting, dan SVM, bahwa dari hasil perbandingan prediksi serta visualisasi error pada data train dan test, masing-masing model menunjukkan performa yang bervariasi.

Berdasarkan grafik perbandingan nilai error (MSE) pada data test, *model Boosting* **memiliki nilai error yang paling kecil**, menandakan bahwa model ini paling stabil dan memiliki kemampuan generalisasi yang baik. Kemudian, *model KNN* **memiliki nilai error yang sedikit lebih tinggi** dibandingkan Boosting. Sementara itu, *model Random Forest* **memiliki nilai error lebih tinggi lagi, menunjukkan adanya overfitting karena selisih yang signifikan antara error data train dan test.**

Sedangkan model SVM menunjukkan nilai error yang paling tinggi di antara keempat model, baik pada data train maupun test, yang mengindikasikan bahwa SVM cenderung underfitting pada data ini.

## Referensi 
https://www.ijraset.com/research-paper/ensemble-models-for-analyzing-students-key-performance-factors Regression and Ensemble Models for Analyzing Students' Key Performance Factors.<br>
https://github.com/samsohail/Students-Performance-Prediction?utm_source=chatgpt.com 
