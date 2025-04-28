# Laporan Proyek Machine Learning - Merri Putri Panggabean

## 1. Domain Proyek

### 1.1 Latar Belakang

Pendidikan merupakan salah satu aspek penting dalam pembangunan sumber daya manusia. Dalam proses pembelajaran, performa siswa sering kali menjadi indikator utama keberhasilan sistem pendidikan. namun, perfoma ini tidak hanya dipengaruhi oleh kemampuan akademik semata, tetapi juga oleh berbagai faktor lain seperti latar belakang keluarga, kondisi sosial-ekonomi, serta ketersediaan fasilitas belajar.
  
## 1.2 Business Understanding

Dalam dunia pendidikan, memahami faktor-faktor yang memengaruhi performa siswa sangat krusial untuk meningkatkan kualitas pembelajaran dan mencegah ketimpangan hasil akademik. Melalui pendekatan analisis data dan machine learning, kita dapat mengklarifikasi permasalahan, merumuskan tujuan, serta menyusun solusi prediktif untuk membantu pengambilan keputusan di bidang pendidikan.

### 1.3 Problem Statements

Berdasarkan latar belakang diatas, berikut batasan masalah yang akan diselesaikan dalam proyek ini :
- Apakah terdapat perbedaan performa akademik berdasarkan jenis kelamin siswa?
- Apakah mengikuti kursus persiapan ujian berdampak signifikan terhadap skor ujian siswa?
- Bisakah kita memprediksi performa siswa (skor ujian) berdasarkan atribut demografisnya?

### 1.4 Goals

Menjawab batasan masalah yang telah dirangkum dan akan menjawab pertanyaan-pertanyaan diatas, berikut tujuan masalahnya :
- Kita perlu untuk menganalisis perbedaan nilai antar siswa berdasarkan jenis kelamin.
- Kita perlu untuk mengetahui pengaruh kursus persiapan ujian terhadap performa ujian siswa.
- Kita perlu untuk membangun model prediksi performa siswa berdasarkan atribut demografisnya, seperti gender,lunch,tingkat pendidikan orang tua dan status kursus.

### 1.5 Solution Statment

untuk mencapai tujuan diatas, maka kita perlu melakukan pendekatan pemodelan yang dimana kita akan menggunakan algoritma seperti :
- Random Forest (RF) : untuk menangani banyak fitur seperti kolom numerik dan kategorial.
- K-Nearest Neighbors (KNN) : melakukan pertimbangan kesamaan antar siswa dalam prediksi skor.
- Boosting : untuk meningkatkan akurasi prediksi dengan menggabungkan beberapa model lemah menjadi kuat.
- Support Vector Machine (SVM) : untuk menangani masalah klasifikasi dengan melihat perbandingan skor tinggi dan rendah.

**Menggunakan MSE (Mean Absolute Error) pada model yang akan di evaluasi.**

## 2. Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi tentang performa akademik siswa berdasarkan beberapa faktor demografis dan sosial, seperti jenis kelamin, kelompok etnis, tingkat pendidikan orang tua, status mengikuti kursus persiapan ujian, dan jenis makan siang yang dikonsumsi. Selain itu, dataset ini juga mencatat skor ujian siswa dalam tiga mata pelajaran utama, yaitu **matematika (math score), membaca (reading score), dan menulis (writing score).**

link dataset : https://www.kaggle.com/datasets/spscientist/students-performance-in-exams 

Dataset mentah yang digunakan dalam proyek ini memiliki 1000 baris data dan 8 kolom. Kolom - kolom tersebut terdiri dari 5 kolom kategori dan 3 kolom numerik. untuk penjelasan mengenai variabel-variabel pada Performance Students dataset adalah sebagai berikut:
- gender : jenis kelamin tiap siswa.
- lunch : jenis makan siang yang setia hari di konsumsi tiap siswa.
- race/ethnicity : jenis kelompok etnis tiap siswa. yang sering dikategorikan dari group A sampai group E.
- parental level of education : tingkat pendidikan orang (misalnya: high school, bachelor’s degree).
- test preparation course : status mengikuti kursus persiapan ujian.
- math score : nilai skor ujian matematika.
- reading score : nilai skor ujian membaca.
- writing score : nilai skor ujian menulis.

## 2.1 Langkah - Langkah Pemrosesan data
A. memasukkan dataset kedalam dataframe menggunakan library pandas.<br>
B. menampilkan informasi dari dataset.<br>
C. menampilkan jumlah data statistik pada dataset.<br>
D. menampilkan jumlah missing value dan duplikat data pada dataset.<br>
E. menangani outlier.<br>
F. menampilkan visualisasi antar fitur numerik antar kategorial.<br>

 ## 2.1.A Membuat dataframe dengan library pandas.
Pada proyek ini, menggunakan fungsi **.read** untuk memasukkan dataset PerformanceStudents.csv kedalam bentuk dataframe menggunakan library pandas dan dataframe yang tersimpan ialah **insu_df**. setelah itu, untuk menampilkan 10 data pertama maka menggunakan **.head(10)**. maka berikut tampilannya :

![alt text](./asset/head.png)

Gambar 1. tampilan 10 data pertama pada dataset insu_df.

## 2.1.B Menampilkan informasi dari dataset.
Pada proyek ini,untuk memahami semua atribut-atribut yang dipakai, memakai fungsi library python yaitu .info() untuk menampilkan atribut-atribut dan baris serta kolom pada dataset. dimana perintah .info() berfungsi untuk menampilkan semua tipe data pada masing-masing atribut dalam dataset.

![alt text](./asset/info.png)

gambar 2. tampilan informasi pada dataset insu_df.

dari output pada gambar diatas, maka dilihat bahwa :
1. terdapat 3 kolom numerik yaitu **math score, writing score dan reading score**.
2. terdapat 5 kolom kategori yaitu **test preparation course, race/ethnicity, parental level of education, lunch dan age**.
3. memiliki jumlah 1000 baris dan 8 kolom pada dataset.
## 2.1.C Menampilkan data statistik dataset.
Pada proyek ini, menggunakan perintah .describe() untuk menampilkan dan mengetahui statistik dasar dari kolom **math score, reading score, writing score** seperti percentile, mean, standar deviasi, jumlah data, min dan max. maka berikut tampilan tabel 2 :

![alt text](./asset/describe.png)

gambar 3. keluaran statistik pada dataset insu_df menggunakan fungsi .describe(). 

## 2.1.D Menampilkan missing value dan duplikat data pada dataset.
Pada proyek ini, untuk mengetahui data memiliki missing value dan duplikat data, maka menggunakan perintah fungsi .insnull() dan .duplicated() pada dataset insu_df, setelah melakukan perintah tersebut, maka akan menampilkan kolom yang memiliki missing value dan jumlah duplikat. maka beikut tampilannya :

![alt text](./asset/missing.png)<br>
Gambar 4. tampilan missing value.

![alt text](./asset/duplikat.png)<br>
Gambar 5. tampilan duplikat data.

## Exploratory Data Analysis
## 2.1.E Melihat outlier pada dataset.
Pada kasus ini, kita akan melihat outlier dalam setiap kolom pada dataset, outlier sendiri adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari hasil pengamatan lainnya. maka berikut tampilan kolom yang outlier :
![alt text](./asset/sebelum.png)<br>
Gambar 6. tampilan sebelum menangani outlier.<br>
dari gambar diatas, terdapat outlier pada kolom math score,reading score dan writing score. untuk menangani outlier kita dapat menggunakan teknik IQR method. IQR adalah *Interquartile Range*. berikut rumus akan kita pakai :<br>
![alt text](./asset/outlier.png)<br>
setelah melakukan penanganan outlier pada kolom dataset, kita dapat melihat hasil dari penanganan outlier yang telah kita lakukan, maka tampilan hasil penganganan outlier, sebagai berikut :<br>
![alt text](./asset/setelah.png)<br>
Gambar 7. setelah melakukan pengangan outlier.<br>
## 2.1.F Menampilkan distribusi numerik dan kategori
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
  Pada gambar 111 merupakan hasil dari visualisasi yang dilakukan, dapat kita lihat bahwa jenis kelamin female lebih tinggi dari jenis kelamin male.
- Kategori lunch<br>
  ![alt text](./asset/Figure_11.png)<br>
  Gambar 12. Hasil visualisasi kolom **lunch**.<br>
  Pada gambar 12, dapat kita lihat bahwa jenis makan siang yang lebih tinggi ialah *standar* daripada *free/recuded* yang lebih rendah.
- Kategori Parental level of education<br>
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

**Visualisasi korelasi math score terhadap data kategori<br>**
![alt text](./asset/parental.png)<br>
![alt text](./asset/race.png)<br>
![alt text](./asset/test.png)<br>
![alt text](./asset/lunch.png)<br>
![alt text](./asset/gender.png)<br>
Gambar 17. Visualisasi korelasi math score pada data kategori.<br>
Pada gambar 17 merupakan hasil visualisasi korelasi *math score* pada data kategori. dimana yang kita lihat terdapat bar chart relatif sama rata, tetapi ada juga perbedaan yang signifikan yaitu :

- Pada bar chart *parental level of education* yang kita lihat bahwa semakin tinggi tingkat pendidikan orang tua maka nilai matematika tiap siswa lebih tinggi dan semakin rendah tingkat pendidikan orangtua maka nilai matematika tiap siswa lebih rendah. maka hal ini juga mempengaruhi nilai tiap siswa.
- dari bar chart *race/ethinicity* yang kita lihat bahwa group E memiliki rata-rata nilai matematika yang paling tinggi secara signifikan dari kelompok lainnya serta group D memiliki rata-rata lebih tinggi dari group A,B dan C dan group yang paling rendah ialah group A.
- dari bar chart *test preparation course* yang kita lihat bahwa ada perbedaan yang relatif tidak merata, dimana completed memiliki rata-rata nilai matematika yang kebih tinggi dibanding dengan none yang memiliki nilai rata-rata matematika yang tergolong rendah.
- dari bar chart *lunch* yang kita ketahui bahwa jenis makan siang yang lebih tinggi ialah standard dibanding dengan free/educed memiliki nilai rendah.
- dari bar chart *gender* yang kita ketahui bahwa jenis kelamin yang mendominasi nilai matematika ialah male memiliki skor nilai 65-70 dibanding dengan female memiliki nilai skor 60-65 lebih rendah dari male.

**Visualisasi hubungan antar fitur numerik dengan fungsi pairplot.<br>**
![alt text](./asset/pairplot.png)<br>
Gambar 18. Hubungan antar fitur numerik dengan pairplot.<br>
Pada Gambar 18 merupakan visualisasi hubungan antar fitur numerik dengan fungsi pairplot. Pada Gambar 18 terdapat fungsi pairplot dari library seaborn yang menunjukkan relasi pasangan dalam dataset. Dari grafik, kita dapat melihat plot relasi masing-masing fitur numerik pada dataset.

Pada kasus ini, kita akan melihat relasi antara semua fitur numerik yang ada dengan fitur target kita  yaitu *'math score'*. Untuk membacanya, perhatikan fitur pada sumbu y, temukan fitur target *'math score'*, dan lihatlah grafik relasi antara fitur tersebut pada sumbu y dengan semua fitur pada sumbu x.

Pada pola sebaran data grafik pairplot sebelumnya, terlihat adanya korelasi positif yang kuat antara setiap pasangan fitur:

'math score' memiliki korelasi positif yang kuat dengan 'reading score'. Hal ini terlihat dari sebaran titik-titik pada scatter plot yang membentuk pola garis lurus menaik dari kiri bawah ke kanan atas.<br>
'math score' juga memiliki korelasi positif yang kuat dengan 'writing score', ditunjukkan oleh pola sebaran yang serupa.<br>
Demikian pula, 'reading score' menunjukkan korelasi positif yang kuat dengan 'writing score'.<br>
Sebaliknya, tidak terlihat adanya indikasi korelasi yang lemah antar fitur-fitur ini karena sebarannya secara jelas membentuk pola hubungan yang positif.

**Korelasi matriks fitur numeri.<br>**
![alt text](./asset/matrik.png)<br>
Gambar 19. Korelasi matrik fitur numerik.<br>
Pada gambar diatas merupakan hasil korelasi matrik pada fitur numerik, yang dimana diketahui bahwa setiap dalam sel adalah nilai koefisien korelasi pearson antara dua fitur. dimana nilai antara 1 dan -1 menunjukkan korelasi yang kuat sedangkan nilai yang mendekati 0 menunjukkan korelasi yang lemah.<br>
dari hasil visualisasi yang kita ketahui bahwa fitur 'reading score' dan 'writing score' keduanya memiliki hubungan yang positif dengan 'math score'. jadi, fitur 'math score' berkorelasi tinggi dengan kedua fitur tersebut.
## 3. Data Preparation
**3.1 Tahap Preparation :<br>**
A. melakukan fungsi 'Standarisasi' pada data numerik,br>
B. mengubah data kategori pada dataset menjadi 'true' dan 'false' dengan menggunakan One-Hot-Encoding.<br>
C. menggunakan teknik PCA.<br>
D. melakukan data splitting menjadi data latih dan data test.<br>

**A. Standarisasi pada data numerik.<br>**

| index | gender | race/ethnicity | parental level of education | lunch        | test preparation course | math score | reading score | writing score |
|:-----:|:------:|:--------------:|:---------------------------:|:------------:|:-----------------------:|:----------:|:-------------:|:-------------:|
| 0     | female | group B         | bachelor's degree           | standard     | none                    | 0.373174   | 0.168406       | 0.374241       |
| 1     | female | group C         | some college                | standard     | completed               | 0.164871   | 1.453233       | 1.338567       |
| 2     | female | group B         | master's degree             | standard     | none                    | 1.622992   | 1.810130       | 1.682969       |
| 3     | male   | group A         | associate's degree          | free/reduced | none                    | -1.362684  | -0.902283      | -1.692172      |
| 4     | male   | group C         | some college                | standard     | none                    | 0.650912   | 0.596682       | 0.443121       |
| ...   | ...    | ...             | ...                         | ...          | ...                     | ...        | ...            | ...            |
| 995   | female | group E         | master's degree             | standard     | completed               | 1.484123   | 2.095647       | 1.820730       |
| 996   | male   | group C         | high school                 | free/reduced | none                    | -0.321169  | -1.045042      | -0.934487      |
| 997   | female | group C         | high school                 | free/reduced | completed               | -0.529472  | 0.097027       | -0.245683      |
| 998   | female | group D         | some college                | standard     | completed               | 0.095437   | 0.596682       | 0.580882       |
| 999   | female | group D         | some college                | free/reduced | none                    | 0.720346   | 1.167716       | 1.200806       | 
<br>

Tabel 1. Tabel standarisasi.<br>
Pada tabel diatas merupakan hasil dari standarisasi yang telah dilakukan pada data numerik untuk mengubah nilai-nilai angka supaya punya rata-rata (mean) menjadi 0 dan standar deviasi menjadi 1. berikut rumus standarisasi.
![alt text](./asset/rumus_standar.png)<br>
Gambar 20. Rumus standarisasi pada data numerik.

**B. One-Hot -Encoding pada data numerik<br>**

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

**C. Teknik PCA<br>**
PCA (Principal Component Analysis) adalah teknik penurunan dimensi yang bertujuan untuk mengubah sekumpulan data asli dengan banyak variabel yang mungkin berkorelasi menjadi sekumpulan variabel baru. dengan hal itulah, teknik PCA pada kolom **'math score', writing score' dan 'reading score'**. mempertahankan komponen PCA seperti gambar dibawah :<br>
![alt text](./asset/pca.png)<br>
Gambar 21. informasi komponen PCA.<br>
Pada gambar diatas merupakan hasil dari komponen PCA yang kita lakukan dengan menggunakan fungsi *.explain _variance_ratio_* pada kolom yang kita komponenkan menjadi tiga. yang diartikan bahwa 89% informasi pada fitur **math score**, 86% informasi pada fitur yang **readning score**.

**D. Data Splitting<br>**
Pada proyek ini kita akan menggunakan data split untuk membagi fitur target yang akan kita latih selanjutnya. untuk melakukan itu, perlu mengimport library data split yaitu *train_test_split*, kemudian membagi variabel menjadi 2 buah yaitu X yang berfungsi untuk menghapus kolom *ma
## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

