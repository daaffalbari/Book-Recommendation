# Laporan Proyek Machine Learning - Daffa Albari

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Dalam proyek ini akan membahas permasalahan mengenai minat membaca buku masyarakat Indonesia yang berhubungan langsung dengan perkembangan teknologi seperti media sosial dan _e-commerce_ di Indonesia.

<img src="https://github.com/daaffalbari/Book-Recommendation/assets/73302268/d8c6679a-4c04-416a-bb68-8588da2b826a" alt="E-Book Illustration" title="E-Book Illustration" width="100%">

Dalam beberapa dekade terakhir, perkembangan teknologi dan juga internet sangat mempengaruhi minat membaca buku masyarakat Indonesia. Contohnya adalah hadirnya media sosial, dan platform seperti YouTube, Netflix, Amazon, dan layanan web sejenis yang lainnya membuat masyarakat lebih banyak menyukai menggunakan platform tersebut sebagai hiburan dan sumber untuk memperoleh informasi. Berbeda pada zaman dahulu di mana masyarakat Indonesia masih bergantung pada media cetak contohnya adalah buku untuk memperoleh ilmu dan informasi.

Membaca sebagai salah satu dari empat keterampilan dalam berbahasa, merupakan suatu proses yang dilakukan serta digunakan oleh pembaca untuk memperoleh pesan yang ingin disampaikan oleh penulis melalui media kata-kata atau bahasa tulis. [[1]](https://core.ac.uk/display/287170379?utm_source=pdf&utm_medium=banner&utm_campaign=pdf-decoration-v1 'Permasalahan Budaya Membaca di Indonesia (Studi Pustaka Tentang Problematika & Solusinya)') Selain itu, membaca juga memerlukan sebuah kemampuan untuk memahami apa yang dibaca sehingga dapat menemukan pesan implisit, dan dapat mengaplikasikan pengetahuan yang terdapat dalam teks melalui proses sintesis dari berbagai gagasan dan informasi. [[2]](#referensi 'Pengajaran Membaca di Sekolah Dasar')

Menurut data yang diambil dari picodi.com, terjadi peningkatan paling tinggi terhadap permintaan jumlah buku di toko buku _online_ pada bulan Desember, yaitu sebesar 12% dari semua transaksi tahunan. Sedangkan penurunan yang paling signifikan terjadi pada bulan Juni sebesar 6% dari semua transaksi tahunan. Berdasarkan survei tentang bagaimana cara pelanggan yang membeli buku, sebanyak 47% membeli buku di toko buku konvensional, sekitar 37% meminjamnya dari perpustakaan, dan hanya 12% yang meminjam dari teman. Sedangkan 10% menyatakan bahwa tidak banyak membaca atau tidak tertarik pada buku. Mengingat semakin populernya teknologi dan internet, ternyata sebesar 55% responden lebih suka memesan buku secara _online_, sedangkan yang lebih memilih untuk membeli buku di toko buku konvensional sebesar 73%. [[3]](https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia 'Pembelian Buku di Indonesia (dan di seluruh Dunia)')

Salah satu situs media sosial yang populer untuk membaca dan menuliskan cerita, yaitu [Wattpad](https://www.wattpad.com 'Wattpad - Where stories live'). Wattpad merupakan _platform_ untuk membuat komunitas membaca dan menghapus penghalang antara pembaca dan penulis. Menurut opini dari Restu I. Aji, dilansir dari Quora.com, beberapa karya yang terdapat pada Wattpad bahkan lebih berwarna dari pada buku yang terbit secara konvensional atau cetak. Menurutnya, tidak sedikit buku yang populer di Wattpad yang justru dialih-mediakan ke buku fisik dan hadir di toko buku _offline_ disertai embel-embel "telah dibaca sekian ratus ribu/juta kali di Wattpad". [[4]](https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline 'Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?')

Dari permasalahan dan latar belakang di atas, maka di dalam proyek ini akan dibuat sebuah model _machine learning_ berupa _recommendation system_ atau sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna. Model ini nantinya dapat digunakan dan di-_deploy_ untuk keperluan tertentu, misalnya diterapkan di dalam katalog buku, daftar buku perpustakaan, media sosial seperti Wattpad, ataupun pada _e-commerce_ yang menjual buku baik digital maupun cetak.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:

1. Bagaimana cara melakukan tahap persiapan data buku, pengguna, dan _rating_ atau penilaian agar dapat digunakan sebagai informasi untuk membuat model _machine learning_ sistem rekomendasi?
2. Bagaimana cara membuat model _machine learning_ untuk sistem rekomendasi buku?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:

1. Melakukan tahap persiapan data sehingga data siap digunakan pada model _machine learning_ untuk sistem rekomendasi.
2. Membuat model _machine learning_ untuk sistem rekomendasi buku terbaik kepada pengguna.

### Solution Statements

Berdasarkan tujuan dari proyek yang telah dipaparkan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:

1. Tahap pra-pemrosesan data atau _data preprocessing_ merupakan tahap untuk mengubah data mentah atau _raw data_ menjadi data yang bersih atau _clean data_ yang siap untuk digunakan pada proses selanjutnya. Tahap ini dapat dilakukan dengan cara, yaitu:
   - Menggabungkan data yang terpisah sehingga dapat digunakan pada tahap selanjutnya.
2. Tahap persiapan data atau _data preparation_ merupakan proses transformasi pada data sehingga data menjadi bentuk yang cocok untuk melakukan proses pemodelan di tahap selanjutnya. Tahap ini dapat dilakukan dengan beberapa teknik, yaitu:
   - Melakukan pengecekan nilai data yang kosong, tidak ada, ataupun _null_ (_missing value_) dan menghapus data tersebut atau mengganti/mengisinya dengan suatu nilai tertentu.
   - Melakukan pengecekan data yang mungkin duplikat agar tidak akan mengganggu hasil dari pemodelan dan sistem yang telah dibangun.
3. Tahap pembuatan model _machine learning_ untuk sistem rekomendasi buku adalah jenis algoritma sistem rekomendasi yang terpersonalisasi atau _personalized recommender system_. Pembuatan model akan menggunakan dua (2) pendekatan, yaitu _content-based filtering recommendation_, dan pendekatan _collaborative filtering recommendation_.
   - **Content-based Filtering Recommendation**  
     Sistem rekomendasi yang berbasis konten (_content-based filtering_) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. _Content-based filtering_ akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan _content-based filtering_ akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
     - TF-IDF Vectorizer  
       Algoritma Term Frequency Inverse Document Frequency Vectorizer (TF-IDF Vectorizer) adalah algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model _machine learning_. [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 'TfidfVectorizer - sci-kit Documentation')

       Kelebihan dari teknik ini adalah tidak membutuhkan data yang diperoleh dari pengguna lain karena rekomendasi yang akan diberikan akan spesifik hanya untuk pengguna tersebut. Sedangkan kekurangan dengan menggunakan teknik ini ialah hasil rekomendasi yang hanya terbatas dari pengguna itu saja dan tidak dapat memperluas data dari penilaian pengguna lain. TF-IDF dapat dihitung menggunakan rumus sebagai berikut: [[6]](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530 'TF-IDF Simplified - Towards Data Science')

       $$idf_i=log \left( \frac{n}{df_i} \right)$$

       Di mana $idf_i$ merupakan skor IDF untuk _term_ $i$; $df_i$ adalah jumlah dokumen yang mengandung _term_ $i$; dan $n$ adalah jumlah total dokumen. Semakin tinggi nilai $df$ suatu _term_, maka semakin rendah $idf$ untuk _term_ tersebut. Ketika jumlah $df$ sama dengan $n$ yang berarti istilah/_term_ tersebut muncul di semua dokumen, $idf$ akan menjadi 0, karena $log(1)=0$.

       Sedangkan nilai TF-IDF merupakan perkalian dari matriks frekuensi _term_ dengan IDF-nya.

       $$w_{i,j}=tf_{i,j} \times idf_i$$

       Di mana $w_{i,j}$ merupakan skor TF-IDF untuk _term_ $i$ pada dokumen $j$; $tf_{i,j}$ adalah frekuensi _term_ untuk _term_ $i$ pada dokumen $j$, dan $idf_i$ adalah skor $idf$ untuk _term_ $i$.

     - Cosine Similarity  
       Teknik _cosine similarity_ digunakan untuk melakukan perhitungan derajat kesamaan (_similarity degree_) antara dua sampel. [[7]](https://www.sciencedirect.com/topics/computer-science/cosine-similarity 'Cosine Similarity - ScienceDirect Topics')

       $$S_c(A,B)=cos(\theta)= \frac{A \times B}{\|A\| \|B\|} = \frac{\displaystyle\sum^{n}_{i=1} A_iB_i}{\sqrt{\displaystyle\sum^{n}_{i=1} A^{2}_{i} } \sqrt{\displaystyle\sum^{n}_{i=1} B^{2}_{i}} }$$

       Di mana $A_i$ dan $B_i$ merupakan komponen dari masing-masing vektor A dan B.
   - **Collaborative Filtering Recommendation**  
     Sistem rekomendasi yang berbasis penyaringan kolaboratif (_collaborative filtering_) adalah sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan preferensi pengguna di masa lalu berdasarkan riwayat pengguna lain yang memiliki preferensi yang sama, misalnya berdasarkan penilaian atau _rating_ yang telah diberikan pengguna di masa lalu. [[8]](https://realpython.com/build-recommendation-engine-collaborative-filtering 'Build a Recommendation Engine With Collaborative Filtering - Real Python') Namun, teknik ini memilki kekurangan yaitu, tidak dapat memberikan rekomendasi item yang tidak memiliki riwayat penilaian/_rating_ atau transaksi.

     Menggunakan teknik _collaborative filtering recommendation_ akan memerlukan proses penyandian (_encoding_) fitur-fitur yang terdapat pada _dataset_ ke dalam bentuk indeks integer, lalu memetakannya ke dalam _dataframe_ yang berkaitan. Kemudian akan dilakukan pembagian distribusi **dataset** dengan rasio tertentu untuk memisahkan data latih (_training data_) dan juga data uji (_validation data_) sebelum dilakukan tahap pemodelan.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Understanding

[<img src="https://github.com/daaffalbari/Book-Recommendation/assets/73302268/93302e30-7975-41e4-af63-a86126b34c10" alt="Book Recommendation Kaggle Dataset" title="Book Recommendation Kaggle Dataset" width="100%">](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Data yang digunakan dalam proyek ini adalah _dataset_ yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang _dataset_ yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 'Build state-of-the-art models for book recommendation system') |
| _Usability_             | 10.00                                                                                                                                                                              |
| Lisensi                 | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0 'Creative Common - CC0 1.0 Universal')                                                                      |
| Penilaian/_Rating_      | Silver                                                                                                                                                                             |
| Jenis dan Ukuran Berkas | zip (25 MB)                                                                                                                                                                        |
| Kategori                | Literature, Art, Culture and Humanities                                                                                                                                            |

Dalam dataset tersebut berisi tiga (3) berkas CSV ([Comma-separated Values](https://en.wikipedia.org/wiki/Comma-separated_values 'Wikipedia - Comma-separated values')), yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut atau fitur sebagai berikut,

|   | ISBN       | Book-Title           | Book-Author          | Year-Of-Publication | Publisher               | Image-URL-S                                       | Image-URL-M                                       | Image-URL-L                                       |
|---|------------|----------------------|----------------------|---------------------|-------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| 0 | 0195153448 | Classical Mythology  | Mark P. O. Morford   | 2002                | Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... |
| 1 | 0002005018 | Clara Callan         | Richard Bruce Wright | 2001                | HarperFlamingo Canada   | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... |
| 2 | 0060973129 | Decision in Normandy | Carlo D'Este         | 1991                | HarperPerennial         | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... |

  - `ISBN` : _International Standard Book Number_
  - `Book-Title` : Judul buku
  - `Book-Author` : Penulis buku
  - `Year-Of-Publication` : Tahun terbit buku
  - `Publisher` : Penerbit buku
  - `Image-URL-S` : Tautan sampul buku ukuran kecil
  - `Image-URL-M` : Tautan sampul buku ukuran sedang
  - `Image-URL-L` : Tautan sampul buku ukuran besar

- **Ratings.csv**, memiliki atribut atau fitur sebagai berikut,

|   | User-ID | ISBN       | Book-Rating |
|---|---------|------------|-------------|
| 0 | 276725  | 034545104X | 0           |
| 1 | 276726  | 0155061224 | 5           |
| 2 | 276727  | 0446520802 | 0           |
| 3 | 276729  | 052165615X | 3           |
| 4 | 276729  | 0521795028 | 6           |


  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `ISBN` : _International Standard Book Number_
  - `Book-Rating` : _Rating_ buku yang diberikan pengguna

- **Users.csv**, memiliki atribut atau fitur sebagai berikut,

|   | User-ID | Location                           | Age  |
|---|---------|------------------------------------|------|
| 0 | 1       | nyc, new york, usa                 | NaN  |
| 1 | 2       | stockton, california, usa          | 18.0 |
| 2 | 3       | moscow, yukon territory, russia    | NaN  |
| 3 | 4       | porto, v.n.gaia, portugal          | 17.0 |
| 4 | 5       | farnborough, hants, united kingdom | NaN  |


  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

Deskripsi statistik untuk _dataset_ `ratings` pada fitur `Book-Rating` dapat dilihat pada gambar di bawah ini.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Preprocessing

Pada tahap pra-pemrosesan data atau _data preprocessing_ dilakukan untuk mengubah data mentah (_raw data_) menjadi data yang bersih (_clean data_) yang siap untuk digunakan pada proses selanjutnya. Ada beberapa tahap yang dilakukan pada _data preprocessing_, yaitu:

- **Menggabungkan Data ISBN**  
  Penggabungan data ISBN buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan _library_ [`numpy`](https://numpy.org). Data ISBN terdapat pada _dataframe_ buku dan _dataframe_ _rating_, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `isbn`.
- **Menggabungkan Data User**  
  Penggabungan data `user_id` buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan _library_ [`numpy`](https://numpy.org). Data `user_id` terdapat pada _dataframe_ _rating_ dan _dataframe_ _user_, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `user_id`.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Preparation

Pada tahap persiapan data atau _data preparation_ dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan nantinya. Ada beberapa tahap yang dilakukan pada _data preparation_, yaitu:

- **Pembagian Dataset**
  
Tahap ini dilakukan pengacakan dataframe ratings terlebih dahulu, lalu kemudian membagi data dengan rasio 80:20, di mana 80% untuk data latih (training data) dan 20% sisanya adalah untuk data uji (validation data).

- **Pengecekan _Missing Value_**  
  Proses pengecekan data yang kosong, hilang, _null_, atau _missing value_ ditemukan pada _dataframe_ `books`, sehingga data yang _missing_ tersebut dihapus atau di-_drop_.

  Sedangkan pada _dataframe_ `ratings` tidak ditemukan adanya _missing value_.

  Kemudian pada _dataframe_ `Users`, terdapat sebanyak 110.762 _missing value_ pada fitur umur. Sehingga data tersebut dapat diganti atau diisi dengan nilai mean atau nilai yang paling sering muncul dalam data umur tersebut.

![image](https://github.com/daaffalbari/Book-Recommendation/assets/73302268/a9fb81b3-cbc3-4b51-9423-5a51382c4026)


- **Pengecekan Data Duplikat**  
  Melakukan pengecekan data duplikat atau data yang sama pada masing-masing _dataframe_. Hasilnya tidak ada data yang duplikat dari ketiga _dataframe_.
- **Data Buku dan Rating**  
  Melakukan penggabungan atau _merge_ data buku dan _rating_ menjadi sebuah _dataframe_.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Modeling

Tahap selanjutnya adalah proses _modeling_ atau membuat model _machine learning_ yang dapat digunakan sebagai sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna dengan beberapa algoritma sistem rekomendasi tertentu.

Berdasarkan tahap pemahaman data atau [_data understanding_](#data-understanding 'Data Understanding') sebelumnya, dapat dilihat bahwa data untuk masing-masing _dataframe_, yaitu data buku, _rating_, dan _users_ tergolong data yang cukup banyak, mencapai ratusan hingga jutaan data. Hal tersebut akan berdampak pada biaya yang akan diperlukan untuk melakukan proses pemodelan _machine learning_, seperti memakan waktu yang lama dan _resource_ RAM ataupun GPU yang cukup besar. Oleh karena itu, dalam kasus ini data yang akan digunakan untuk proses pemodelan _machine learning_ data akan dibatasi hanya 10.000 baris data buku dan 7500 baris data _rating_.


1. **Content-based Recommendation**

   - TF-IDF Vectorizer  
     TF-IDF Vectorizer akan mentransformasikan teks menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 5.575 data _author_ atau penulis buku.

|                                        | quin | ry  | coen | katie | company | blume | lustbader | hawks | napier | cordingly | cast | cash | kindred | sobel | sachar | clegg | mcalister | candice | limbaugh | stern |
|----------------------------------------|------|-----|------|-------|---------|-------|-----------|-------|--------|-----------|------|------|---------|-------|--------|-------|-----------|---------|----------|-------|
| Book-Title                             |      |     |      |       |         |       |           |       |        |           |      |      |         |       |        |       |           |         |          |       |
| The Very Hungry Caterpillar Board Book | 0.0  | 0.0 | 0.0  | 0.0   | 0.0     | 0.0   | 0.0       | 0.0   | 0.0    | 0.0       | 0.0  | 0.0  | 0.0     | 0.0   | 0.0    | 0.0   | 0.0       | 0.0     | 0.0      | 0.0   |
| Technical Writing                      | 0.0  | 0.0 | 0.0  | 0.0   | 0.0     | 0.0   | 0.0       | 0.0   | 0.0    | 0.0       | 0.0  | 0.0  | 0.0     | 0.0   | 0.0    | 0.0   | 0.0       | 0.0     | 0.0      | 0.0   |
| Five Children and It                   | 0.0  | 0.0 | 0.0  | 0.0   | 0.0     | 0.0   | 0.0       | 0.0   | 0.0    | 0.0       | 0.0  | 0.0  | 0.0     | 0.0   | 0.0    | 0.0   | 0.0       | 0.0     | 0.0      | 0.0   |
| Rose Madder                            | 0.0  | 0.0 | 0.0  | 0.0   | 0.0     | 0.0   | 0.0       | 0.0   | 0.0    | 0.0       | 0.0  | 0.0  | 0.0     | 0.0   | 0.0    | 0.0   | 0.0       | 0.0     | 0.0      | 0.0   |
| Becoming a Thinking Christian          | 0.0  | 0.0 | 0.0  | 0.0   | 0.0     | 0.0   | 0.0       | 0.0   | 0.0    | 0.0       | 0.0  | 0.0  | 0.0     | 0.0   | 0.0    | 0.0   | 0.0       | 0.0     | 0.0      | 0.0   |

   - _Cosine Similarity_  
     _Cosine Similarity_ akan melakukan perhitungan derajat kesamaan (_similarity degree_) antar judul buku. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 10.000 data buku juga.

| Book-Title           | Classical Mythology | Clara Callan | Decision in Normandy | Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It | The Mummies of Urumchi | The Kitchen God's Wife | What If?: The World's Foremost Military Historians Imagine What Might Have Been | PLEADING GUILTY | Under the Black Flag: The Romance and the Reality of Life Among the Pirates | Where You'll Find Me: And Other Stories | ... | Kaplan GRE Exam 2004 | Around the World in Eighty Days (Tor Classics) | The Exploits of the Incomparable Mulla Nasrudin / The Subtleties of the Inimitable Mulla Nasrudin | The Hours : A Novel | Beloved (Penguin Great Books of the 20th Century) | Read This and Tell Me What It Says : Stories (Bard Book) | The Star Rover | Die Keltennadel. | Tod in der Datscha. | Dunkel. |
|----------------------|---------------------|--------------|----------------------|----------------------------------------------------------------------------------------------------|------------------------|------------------------|---------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------|-----------------------------------------|-----|----------------------|------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------|---------------------------------------------------|----------------------------------------------------------|----------------|------------------|---------------------|---------|
| Book-Title           |                     |              |                      |                                                                                                    |                        |                        |                                                                                 |                 |                                                                             |                                         |     |                      |                                                |                                                                                                   |                     |                                                   |                                                          |                |                  |                     |         |
| Classical Mythology  | 1.0                 | 0.0          | 0.0                  | 0.0                                                                                                | 0.0                    | 0.0                    | 0.0                                                                             | 0.0             | 0.0                                                                         | 0.0                                     | ... | 0.0                  | 0.0                                            | 0.0                                                                                               | 0.0                 | 0.0                                               | 0.0                                                      | 0.0            | 0.0              | 0.0                 | 0.0     |
| Clara Callan         | 0.0                 | 1.0          | 0.0                  | 0.0                                                                                                | 0.0                    | 0.0                    | 0.0                                                                             | 0.0             | 0.0                                                                         | 0.0                                     | ... | 0.0                  | 0.0                                            | 0.0                                                                                               | 0.0                 | 0.0                                               | 0.0                                                      | 0.0            | 0.0              | 0.0                 | 0.0     |
| Decision in Normandy | 0.0                 | 0.0          | 1.0                  | 0.0                                                                                                | 0.0                    | 0.0                    | 0.0                                                                             | 0.0             | 0.0                                                                         | 0.0                                     | ... | 0.0                  | 0.0                                            | 0.0                                                                                               | 0.0                 | 0.0                                               | 0.0                                                      | 0.0            | 0.0              | 0.0                 | 0.0     |


   - Hasil _Top-N Recommendation_
     Hasil pengujian sistem rekomendasi dengan pendekatan _content-based recommendation_ adalah sebagai berikut.

|   | Book-Title                                        | Book-Author          |
|---|---------------------------------------------------|----------------------|
| 0 | Isola Dell Caduto                                 | Carlo Lucarelli      |
| 1 | Un Giorno Dopo L'altro                            | Carlo Lucarelli      |
| 2 | Almost blue (Stile libero)                        | Carlo Lucarelli      |
| 3 | Tecniche di seduzione                             | Andrea De Carlo      |
| 4 | Due di due (Bestsellers)                          | Andrea De Carlo      |
| 5 | Kissing the Witch: Old Tales in New Skins         | Emma Donoghue        |
| 6 | The DREAMS OUR STUFF IS MADE OF: How Science F... | Thomas M. Disch      |
| 7 | The Bookshop : A Novel                            | Penelope Fitzgerald  |
| 8 | Cruel As the Grave                                | Sharon Kay Penman    |
| 9 | Naked Lunch                                       | William S. Burroughs |


     Dapat dilihat bahwa sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul buku berdasarkan input atau masukan sebuah judul buku, yaitu "Decision in Normandy", dan diperoleh beberapa judul buku yang berdasarkan perhitungan sistem.

2. **Collaborative Filtering Recommendation**

   - Data Preparation  
     _Data preparation_ yang dilakukan adalah dengan melakukan penyandian (_encoding_) fitur `user_id` dan `isbn` pada _dataframe_ `ratings` ke dalam bentuk indeks integer. Kemudian melakukan pemetaan fitur yang telah di-_encoding_ tersebut ke dalam masing-masing _dataframe_ yang `ratings`.

     Diperoleh jumlah _user_ sebesar 1204, jumlah buku sebesar 4565, nilai minimal _rating_ yaitu 1, dan nilai maksimum _rating_ yaitu 10.

   - Split Training Data dan Validation Data  
     Tahap ini dilakukan pengacakan _dataframe ratings_ terlebih dahulu, lalu kemudian membagi data dengan rasio 80:20, di mana 80% untuk data latih (_training data_) dan 20% sisanya adalah untuk data uji (_validation data_).


      |      | User-ID | ISBN       | Book-Rating | user | book |
      |------|---------|------------|-------------|------|------|
      | 4362 | 278418  | 0060186534 | 0           | 678  | 4063 |
      | 7083 | 278418  | 0590337785 | 0           | 678  | 6681 |
      | 2346 | 277565  | 1862082626 | 0           | 308  | 2264 |
      | 5413 | 278418  | 0373163177 | 0           | 678  | 5087 |
      | 2266 | 277523  | 0446613843 | 10          | 288  | 386  |
      | ...  | ...     | ...        | ...         | ...  | ...  |
      | 5338 | 278418  | 0373114109 | 0           | 678  | 5015 |
      | 5354 | 278418  | 0373114443 | 0           | 678  | 5031 |
      | 3746 | 278144  | 0679459596 | 0           | 567  | 3525 |
      | 869  | 277107  | 081297106X | 10          | 144  | 859  |
      | 3980 | 278188  | 1551665344 | 7           | 587  | 3737 |

   - Model Development dan Hasil  
     Berdasarkan model yang telah di-_training_, berikut adalah hasil pengujian sistem rekomendasi buku dengan pendekatan _collaborative filtering recommendation_.

| Book with high ratings from user                                                                            |
|-------------------------------------------------------------------------------------------------------------|
| ----------------------------------------                                                                    |
| ========================================                                                                    |
| Top 10 Books Recommendation                                                                                 |
| ----------------------------------------                                                                    |
| Rebecca : Daphne Du Maurier                                                                                 |
| Girl with a Pearl Earring : Tracy Chevalier                                                                 |
| The Secret Life of Bees : Sue Monk Kidd                                                                     |
| Life of Pi : Yann Martel                                                                                    |
| The Girl Who Loved Tom Gordon : A Novel : Stephen King                                                      |
| A Cold Heart: An Alex Delaware Novel : JONATHAN KELLERMAN                                                   |
| The Golden Mean: In Which the Extraordinary Correspondence of Griffin &amp; Sabine Concludes : Nick Bantock |
| Politically Correct Bedtime Stories: Modern Tales for Our Life and Times : James Finn Garner                |
| The Watsons Go to Birmingham - 1963 (Yearling Newbery) : CHRISTOPHER PAUL CURTIS                            |
| Oceano Mare : Alessandro Baricco                                                                            |


     Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `user_id` **278418**. Lalu akan dicari buku dengan _rating_ terbaik dari user tersebut, yaitu,

     - **Rebecca ** oleh **Daphne Du Maurier**
     - **Girl with a Pearl Earring** oleh **Tracy Chevalier**

     Kemudian sistem akan membandingkan antara buku dengan _rating_ tertinggi dari _user_ dan semua buku, kecuali buku yang telah dibaca tersebut, lalu akan mengurutkan buku yang akan direkomendasikan berdasarkan nilai rekomendasi yang tertinggi. Dapat dilihat terdapat 10 daftar buku yang direkomendasikan oleh sistem.

     Dapat dibandingkan antara **_Book with high ratings from user_** dan **_Top 10 Books Recommendation_**, terdapat buku dengan penulis atau _author_ yang sama, yaitu **The Kitchen God's Wife** oleh **Amy Tan**. Dengan begitu, dapat dikatakan bahwa sistem yang telah dibangun dapat merekomendasikan buku kepada pengguna dengan prediksi yang cukup sesuai.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Evaluation

1. **Content-based Recommendation**  
   Pada tahap evaluasi untuk model sistem rekomendasi dengan pendekatan _content-based recommendation_ dapat menggunakan evaluasi dengan metrik akurasi yang diperoleh dari,

   $$Accuracy=\frac{\displaystyle\sum_{i=1}^{n} RecommendedBooks_i}{\displaystyle\sum_{i=1}^{n} BooksWithSameAuthor_i} \times 100$$

   Masih menggunakan data yang sama pada tahap [Modeling](#modeling 'Modeling') _content-based recommendation_, pada proses Hasil _Top-N Recommendation_, yaitu penulis buku atau `book_author` Toni Morrison, akan dilakuakn proses pencarian jumlah judul buku atau `book_title` dengan penulis atau _author_ yang sama. Pencarian tersebut menggunakan variabel baru yang di mana akan mengambil sebuah data buku yang telah dibaca oleh pengguna. Hasil yang diperoleh adalah Toni Morrison memiliki jumlah buku sebanyak 7 buah buku.

   Proses perhitungan akurasi dilakukan dengan membagi banyaknya rekomendasi buku yang dihasilkan, dibagi dengan banyaknya jumlah buku yang ditulis oleh _author_ atau penulis yang sama, kemudian dikalikan dengan 100. Sehingga diperoleh nilai **akurasi** sebesar **57.14%**.

2. **Collaborative Filtering Recommendation**  
   Berdasarkan model _machine learning_ yang sudah dibangun menggunakan _embedding layer_ dengan _Adam optimizer_ dan _binary crossentropy loss function_, metrik yang digunakan adalah _Root Mean Squared Error_ (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,

   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$

   Di mana, nilai $n$ merupakan jumlah _dataset_, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam _dataset_.

   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati.

   Berikut merupakan visualisasi hasil _training_ dan _validation error_ dari metrik RMSE serta _training_ dan _validation loss_ ke dalam grafik plot.

![image](https://github.com/daaffalbari/Book-Recommendation/assets/73302268/052d3478-8635-44df-8fae-fdaa199f6e0d)

Dari grafik di atas menunjukan bahwa model yang dibuat sudah bagus atau _Goodfit_.


[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Kesimpulan

Kesimpulannya adalah model yang digunakan untuk melakukan rekomendasi buku berdasarkan teknik _Content-based Recommendation_ dan teknik _Collaborative Filtering Recommendation_ telah berhasil dibuat dan sesuai dengan preferensi pengguna. Dengan adanya model ini diharapkan bisa menyelesaikan masalah-masalah yang telah diuraikan di awal tadi. Pada _collaborative filtering_ diperlukan data _rating_ dari pengguna, sedangkan pada _content-based filtering_, data _rating_ tidak diperlukan karena analisis sistem rekomendasi akan berdasarkan atribut item dari masing-masing buku. Sistem rekomendasi dengan pendekatan _content-based recommendation_ dan _collaborative filtering recommendation_ memiliki kelebihan dan kekurangannya masing-masing.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Referensi

[1] L. Tahmidaten and W. Krismanto, "Permasalahan Budaya Membaca di Indonesia (Studi Pustaka Tentang Problematika & Solusinya)", _Scholaria: Jurnal Pedidikan dan Kebudayaan_, vol. 10, no, 1, pp. 22-23, Jan. 2020, doi: 10.24246/j.js.2020.v10.i1.p22-33, Retrieved from: https://ejournal.uksw.edu/scholaria/article/view/2656.

[2] F. Rahim, _Pengajaran Membaca di Sekolah Dasar_, Jakarta: Sinar Grafika, 2008.

[3] Picodi, "Pembelian Buku di Indonesia (dan di seluruh Dunia)", _Picodi.com_, 2019, Retrieved from: https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia.

[4] "Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?", _Quora_, Retrieved from: https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline.

[5] scikit-learn, "sklearn.feature_extraction.text.TfidfVectorizer", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.

[6] L. Ramadhan, "TF-IDF Simplified, A short introduction to TF-IDF vectorizer", _Towards Data Science_, 2021, Retrieved from: https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530.

[7] ScienceDirect, "Cosine Similarity", Retrieved from: https://www.sciencedirect.com/topics/computer-science/cosine-similarity.

[8] A. Ajitsaria, "Build a Recommendation Engine With Collaborative Filtering", _Real Python_, 2019, Retrieved from: https://realpython.com/build-recommendation-engine-collaborative-filtering.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')
