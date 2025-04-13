import numpy as np
import pandas as pd
import tensorflow as tf
from heuristics import DFT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from printStuff import print_label_statistics

# Read and preprocess labels
file = open("../data/labels.txt", "r")
labels = file.readlines()
labels = [label.strip() for label in labels]

# Read and preprocess company data
file = open("../data/ml_insurance_challenge.csv", "r")
data = pd.read_csv(file)
companies = data.apply(lambda row: {
    'description': row.iloc[0],
    'bussines_tags': row.iloc[1],
    'sector': row.iloc[2],
    'category': row.iloc[3],
    'niche': row.iloc[4]
}, axis=1)

# Get labels using DFT
results = DFT(labels, companies, 0.5)

# Prepare data for neural network
X = []
y = []
for idx, (label, score) in results.items():
    company = companies[idx]
    # Combine all text fields
    text = f"{company['description']} {company['bussines_tags']} {company['sector']} {company['category']} {company['niche']}"
    X.append(text)
    y.append(label)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Tokenize and pad text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=200)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# Build neural network model
model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Function to predict labels for new companies
def predict_company_label(company_text):
    sequence = tokenizer.texts_to_sequences([company_text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]
