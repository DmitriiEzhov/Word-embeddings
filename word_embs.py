import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, concatenate, Dropout, BatchNormalization
import shutil
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split

# Загрузка данных
train_data = pd.read_csv('/content/names_train.csv')
test_data = pd.read_csv('/content/names_test.csv')

# Токенизация имен
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_data['first_name'].tolist() + train_data['first_name_draft'].tolist())

# Преобразование имен в последовательности
first_name_seq_train = tokenizer.texts_to_sequences(train_data['first_name'])
first_name_draft_seq_train = tokenizer.texts_to_sequences(train_data['first_name_draft'])

first_name_seq_test = tokenizer.texts_to_sequences(test_data['first_name'])
first_name_draft_seq_test = tokenizer.texts_to_sequences(test_data['first_name_draft'])

# Паддинг последовательностей
max_len = max(max(len(seq) for seq in first_name_seq_train), max(len(seq) for seq in first_name_draft_seq_train))
first_name_seq_train_padded = pad_sequences(first_name_seq_train, maxlen=max_len, padding='post')
first_name_draft_seq_train_padded = pad_sequences(first_name_draft_seq_train, maxlen=max_len, padding='post')

first_name_seq_test_padded = pad_sequences(first_name_seq_test, maxlen=max_len, padding='post')
first_name_draft_seq_test_padded = pad_sequences(first_name_draft_seq_test, maxlen=max_len, padding='post')

# Объединение данных
X_train = [first_name_seq_train_padded, first_name_draft_seq_train_padded]
X_test = [first_name_seq_test_padded, first_name_draft_seq_test_padded]

# Загрузка предобученных эмбеддингов Word2Vec
word_vectors = api.load("word2vec-google-news-300")

# Функция для генерации эмбеддингов
def generate_embeddings(tokenizer, sequences, max_len, word_vectors):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word_vectors.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
    return embedding_matrix

# Генерация эмбеддингов для словаря токенизатора
embedding_matrix = generate_embeddings(tokenizer, first_name_seq_train + first_name_draft_seq_train, max_len, word_vectors)

# Определение модели Siamese Network
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                            output_dim=word_vectors.vector_size,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)

encoded_1 = embedding_layer(input_1)
encoded_2 = embedding_layer(input_2)

shared_lstm = Bidirectional(LSTM(64, return_sequences=True))
output_1 = shared_lstm(encoded_1)
output_2 = shared_lstm(encoded_2)

merged = concatenate([output_1, output_2], axis=-1)

flat_merged = LSTM(64)(merged)

dropout = Dropout(0.5)(flat_merged)
batch_norm = BatchNormalization()(dropout)
dense = Dense(64, activation='relu')(batch_norm)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Объединение данных для train и val split
combined_X_train = np.hstack((first_name_seq_train_padded, first_name_draft_seq_train_padded))

# Разделение тренировочного набора на тренировочную и валидационную части
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(combined_X_train, y_train, test_size=0.2, random_state=42)

# Разделение обратно на две части для input_1 и input_2
X_train_split_1 = X_train_split[:, :max_len]
X_train_split_2 = X_train_split[:, max_len:]
X_val_split_1 = X_val_split[:, :max_len]
X_val_split_2 = X_val_split[:, max_len:]

# Обучение модели
model.fit([X_train_split_1, X_train_split_2], y_train_split, epochs=30, batch_size=64, validation_data=([X_val_split_1, X_val_split_2], y_val_split))

# Предсказание на тестовом наборе
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Создание файла предсказаний
predictions = pd.DataFrame({
    'Id': test_data['Id'],
    'Category': y_pred.flatten()
})
predictions.to_csv('predictions.csv', index=False)

# Создание архива с предсказаниями
shutil.make_archive('submission', 'zip', '.', 'predictions.csv')
