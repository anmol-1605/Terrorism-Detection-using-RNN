import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

dataset_path = 'terrorism_dataset.csv'
df = pd.read_csv(dataset_path)

# Preprocessing
max_vocab = 3500
max_len = 60

def preprocess_text(text):
    return text.lower()

texts = df['text'].apply(preprocess_text).tolist()
labels = df['label'].values

tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post')

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
bow_features = vectorizer.fit_transform(texts).toarray()

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
y = to_categorical(encoded_labels, num_classes=3)

# Train-test split
X_seq_train, X_seq_val, X_bow_train, X_bow_val, y_train, y_val = train_test_split(
    padded_seq, bow_features, y, test_size=0.2, random_state=42)

# Build model
input_seq = tf.keras.layers.Input(shape=(max_len,))
embedding = Embedding(max_vocab, 48)(input_seq)
lstm = LSTM(32, dropout=0.4, recurrent_dropout=0.2)(embedding)

input_bow = tf.keras.layers.Input(shape=(bow_features.shape[1],))
bow_dense = Dense(24, activation='relu')(input_bow)
bow_out = Dropout(0.4)(bow_dense)

concat = tf.keras.layers.concatenate([lstm, bow_out])
dense1 = Dense(32, activation='relu')(concat)
drop1 = Dropout(0.5)(dense1)
output = Dense(3, activation='softmax')(drop1)

model = tf.keras.models.Model(inputs=[input_seq, input_bow], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Train
history = model.fit(
    [X_seq_train, X_bow_train], y_train,
    epochs=15,
    batch_size=16,
    validation_data=([X_seq_val, X_bow_val], y_val),
    callbacks=callbacks,
    verbose=1
)

# Save everything
model.save('terrorism_detection_model.h5')
with open('tokenizer.pickle', 'wb') as f: pickle.dump(tokenizer, f)
with open('vectorizer.pickle', 'wb') as f: pickle.dump(vectorizer, f)
with open('label_encoder.pickle', 'wb') as f: pickle.dump(encoder, f)

print(f"Model trained and saved. Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
