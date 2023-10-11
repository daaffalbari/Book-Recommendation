# %% [markdown]
# # Proyek Akhir: Membuat Sistem Rekomendasi Buku
# <hr>

# %% [markdown]
# ## Data Diri
# Nama: Daffa Albari <br>
# Email: daffaa.albari@gmail.com <br>
# Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %% [markdown]
# ## Load Data

# %%
books = pd.read_csv('./data/Books.csv')
ratings = pd.read_csv('./data/Ratings.csv')
users = pd.read_csv('./data/Users.csv')

# %%
books.head()

# %%
ratings.head()

# %%
users.head()

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# ### 2.1 Jumlah data dalam masing-masing tabel|

# %%
print('Jumlah data buku: ', len(books['ISBN'].unique()))
print('Jumalah data rating: ', len(ratings['User-ID'].unique()))
print('Jumlah data user: ', len(users['User-ID'].unique()))

# %% [markdown]
# ### 2.2 Univariate Exploratory Data Analysis (EDA)

# %% [markdown]
# Variabel-variabel yang digunakan dalam analisis ini adalah:
# - Books: Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.
# - Users: Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.
# - Ratings: Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

# %% [markdown]
# #### Books Variabel

# %%
books.info()

# %%
print('Banyak Data: ', len(books))

# %% [markdown]
# Berdasarkan output di atas, kita dalam mengetahui bahwa Books.csv memiliki 271360 entri. 

# %%
books.head()

# %%
print('Banyak ISBN: ', len(books['ISBN'].unique()))
print('Banyak Judul: ', len(books['Book-Title'].unique()))
print('Banyak Penulis: ', len(books['Book-Author'].unique()))

# %% [markdown]
# #### Ratings Variabel

# %%
ratings.head()

# %%
ratings.info()

# %%
users.info()

# %%
ratings.describe()

# %%
print('Jumlah user id: ', len(ratings['User-ID'].unique()))
print('Jumlah Book-Rating: ', len(ratings['Book-Rating'].unique()))
print('Jumlah data rating: ', len(ratings))

# %% [markdown]
# #### Users Variabel

# %%
users.describe()

# %%
users.info()

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### 1. Menggabungkan Data Buku

# %%
books_all = np.concatenate((
  books['ISBN'].unique(),
  ratings['ISBN'].unique()
))

books_all = np.sort(np.unique(books_all))
print('Jumlah data buku dan user: ', len(books_all))

# %% [markdown]
# ### 2. Menggabungkan Data User

# %%
users_all = np.concatenate((
  ratings['User-ID'].unique(),
  users['User-ID'].unique()
))

users_all = np.sort(np.unique(users_all))
print('Jumlah data user: ', len(users_all))

# %% [markdown]
# ## Data Preparation
# Pada tahap ini, dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan nantinya. Dalam kasus ini, tahap data preparation dilakukan dengan mengatasi missing value, pengecekan data duplikat, dan penggabungan data buku dan data rating.

# %% [markdown]
# ### 1. Mengatasi Missing Value

# %% [markdown]
# #### Book

# %%
book_df = books

book_df.isnull().sum()

# %% [markdown]
# Terlihat terdapat nilai kosong atau null pada fitur book author, publisher dan juga Image-URL-L. 

# %%
book_df = book_df.dropna()

book_df.isnull().sum()

# %% [markdown]
# #### Ratings

# %%
rating_df = ratings

rating_df.isnull().sum()

# %%
rating_df['Book-Rating'].value_counts().plot(kind='bar')

# %% [markdown]
# Sistem rekomendasi biasanya memiliki data umpan balik implisit, di mana pengguna tidak secara eksplisit memberikan penilaian namun tindakan mereka dapat digunakan sebagai indikasi preferensi. Dalam hal ini, peringkat 0 mungkin mewakili peringkat yang hilang atau tidak diketahui, yang sering kali dianggap sebagai "umpan balik negatif implisit". Artinya, pengguna yang belum memberikan rating apa pun untuk sebuah buku belum tentu menunjukkan bahwa mereka tidak menyukai buku tersebut; mereka mungkin tidak menyatakan pendapat apa pun tentang hal itu.

# %% [markdown]
# #### Users

# %%
user_df = users

user_df.isnull().sum()

# %% [markdown]
# Berdasarkan deskripsi di atas, dapat dilihat bahwa pada dataframe users terdapat atribut yang memiliki nilai kosong atau null, yaitu pada atribut age sebanyak 110.762 data.

# %%
user_df['Age'] = user_df['Age'].fillna(user_df['Age'].mean())
user_df.isnull().sum()

# %% [markdown]
# ### 2. Mengatasi Duplikasi Data

# %%
print('Jumlah Data Buku yang duplikat: ', book_df.duplicated().sum())
print('Jumlah Data Rating yang duplikat: ', rating_df.duplicated().sum())
print('Jumlah Data User yang duplikat: ', user_df.duplicated().sum())

# %% [markdown]
# ### 3. Data Buku dan Data Rating

# %%
books_ratings = pd.merge(book_df, rating_df, on='ISBN')

books_ratings.head()

# %% [markdown]
# ## Model Development dengan Content Based Filtering

# %%
# Batasan data
book_df = book_df[:10000]
rating_df = rating_df[:7500]

# %% [markdown]
# #### TF-IDF Vectorize

# %%
tfidf = TfidfVectorizer()

tfidf.fit(book_df['Book-Author'])


# %%
tfidf_matrix = tfidf.transform(book_df['Book-Author'])
tfidf_matrix.shape

# %%
tfidf_matrix.todense()

# %%
pd.DataFrame(
  tfidf_matrix.todense(),
  columns=tfidf.get_feature_names_out(),
  index=book_df['Book-Title']
).sample(20, axis=1).sample(10, axis=0)

# %% [markdown]
# ### Cosine Similarity

# %%
# Menghitung cosine similarity pada matrix tfidf
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim

# %% [markdown]
# Pada tahapan ini, kita menghitung cosine similarity dataframe tfidf_matrix yang kita peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, kita telah berhasil menghitung kesamaan (similarity) antar restoran. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. 

# %%
cosine_sim_df = pd.DataFrame(
  cosine_sim,
  columns=book_df['Book-Title'],
  index=book_df['Book-Title']
)

print(cosine_sim_df.shape)
cosine_sim_df.head()

# %% [markdown]
# ### Mendapatkan Rekomendasi

# %%
def author_recommendation(title, similarity_data=cosine_sim_df, items=book_df[['Book-Title', 'Book-Author']], k=10):
  index = similarity_data.loc[:,title].to_numpy().argpartition(range(-1, -k, -1))
  closest = similarity_data.columns[index[-1:-(k+2):-1]]
  closest = closest.drop(title, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)
  

# %%
readed_book_title = 'Decision in Normandy'

# %%
author_recommendation(readed_book_title).drop_duplicates()

# %% [markdown]
# Dapat dilihat bahwa sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul buku berdasarkan input atau masukan sebuah judul buku, yaitu "Decision in Normandy", dan diperoleh beberapa judul buku berdasarkan model yang telah dibuat

# %% [markdown]
# ## Model Development dengan Collaborative Filtering Recommendation

# %% [markdown]
# ### Data Preparation

# %%
user_ids = rating_df['User-ID'].unique().tolist()

# Melakukan encoding userID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('Encoded UserID: ', user_to_user_encoded)

# Melakukan proses encoding angka ke user ID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)

# %%
book_ids = rating_df['ISBN'].unique().tolist()

# Melakukan encoding bookID
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}
print('Encoded BookID: ', book_to_book_encoded)

# Melakukan proses encoding angka ke book id
book_encoded_to_book = {i: x for i, x in enumerate(book_ids)}
print(book_encoded_to_book)

# %%
rating_df['user'] = rating_df['User-ID'].map(user_to_user_encoded)
rating_df['book'] = rating_df['ISBN'].map(book_to_book_encoded)

# %%
num_users = len(user_encoded_to_user)
print(num_users)
num_books = len(book_encoded_to_book)
print(num_books)

min_ratings = min(rating_df['Book-Rating'])
max_ratings = max(rating_df['Book-Rating'])

# %% [markdown]
# ### Membagi Data untuk training dan validasi

# %%
rating_df = rating_df.sample(frac=1, random_state=412)
rating_df

# %%
x = rating_df[['user', 'book']].values
y = rating_df['Book-Rating'].apply(lambda x: (x-min_ratings) / (max_ratings-min_ratings)).values

train_indices = int(0.8 * rating_df.shape[0])

X_train, X_val, y_train, y_val = (
  x[:train_indices],
  x[train_indices:],
  y[:train_indices],
  y[train_indices:]
)

print(x, y)

# %%
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError

# %%
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.user_bias      = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_books, 1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0])
        user_bias   = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias   = self.book_bias(inputs[:, 1])
        
        dot_user_book = tf.tensordot(user_vector, book_vector, 2) 
        
        x = dot_user_book + user_bias + book_bias
        
        return tf.nn.sigmoid(x)

# %%
model = RecommenderNet(num_users, num_books, 50)

model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[RootMeanSquaredError()])

# %%
history = model.fit(
  x= X_train,
  y = y_train,
  batch_size=16,
  epochs=30,
  validation_data=(X_val, y_val)
)

# %%
rmse     = history.history['root_mean_squared_error']
val_rmse = history.history['val_root_mean_squared_error']

loss     = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (12, 4))
plt.subplot(1, 2, 1)
plt.plot(rmse,     label='RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.title('Training and Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Root Mean Squared Error')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(loss,     label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

# %% [markdown]
# ### Mendapatkan Rekomendasi

# %%
datasetBook = book_df
datasetRating = rating_df

# %%
datasetBook.head(2)

# %%
datasetRating.head(2)

# %%
userId      = datasetRating['User-ID'].sample(1).iloc[0]
readedBooks = datasetRating[datasetRating['User-ID'] == userId]

notReadedBooks = datasetBook[~datasetBook['ISBN'].isin(readedBooks['ISBN'].values)]['ISBN'] 
notReadedBooks = list(
    set(notReadedBooks).intersection(set(book_to_book_encoded.keys()))
)

notReadedBooks = [[book_to_book_encoded.get(x)] for x in notReadedBooks]
userEncoder    = user_to_user_encoded.get(userId)
userBookArray = np.hstack(
    ([[userEncoder]] * len(notReadedBooks), notReadedBooks)
)

# %%
ratings = model.predict(userBookArray).flatten()

topRatingsIndices   = ratings.argsort()[-10:][::-1]
recommendedBookIds = [
    book_encoded_to_book.get(notReadedBooks[x][0]) for x in topRatingsIndices
]

print('Showing recommendations for users: {}'.format(userId))
print('=====' * 8)
print('Book with high ratings from user')
print('-----' * 8)

topBookUser = (
    readedBooks.sort_values(
        by = 'Book-Rating',
        ascending=False
    )
    .head(5)
    .ISBN.values
)

bookDfRows = datasetBook[datasetBook['ISBN'].isin(topBookUser)]
for row in bookDfRows.itertuples():
    print(row['Book-Title'], ':', row.book_author)

print('=====' * 8)
print('Top 10 Books Recommendation')
print('-----' * 8)

recommended_resto = datasetBook[datasetBook['ISBN'].isin(recommendedBookIds)]
for row in recommended_resto.itertuples():
    print(row[2], ':', row[3])


