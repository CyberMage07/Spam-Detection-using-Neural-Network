import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load the dataset
dataset = pd.read_csv('spam.csv')
sentences = dataset['Message'].tolist()
labels = dataset['Category'].tolist()

# Separate the dataset into training and testing sets
training_size = int(len(sentences) * 0.8)
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# Convert labels into numpy arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Tokenize the sentences
vocab_size = 600
max_length = 60
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# Convert sentences to sequences and pad them
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
num_epochs = 30
history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

# Save the model
#*model.save('spam_detection_model')
model.save('spam_detection_model.h5')


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get the weights of the embedding layer
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]

# Write out the embedding vectors and metadata
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word, index in tokenizer.word_index.items():
    if index < vocab_size:
        vec = weights[index]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

# Function to predict whether a message is spam or not
import tkinter as tk
from tkinter import messagebox

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window

def predict_spam(text_messages):
    global dataset, sentences, labels, training_size, training_sentences, training_labels, training_labels_final

    # Check for specific keywords indicating high likelihood of spam
    keywords = ["winner", "free", "offer", "prize", "urgent", "claim", "cash", "congratulations", "Winner", "Win"]
    for keyword in keywords:
        if keyword.lower() in text_messages.lower():
            messagebox.showwarning("Spam Alert", "The given message contains a keyword often associated with spam. It is classified as SPAM.")
            # Add the new message to the dataset as spam
            dataset = dataset._append({'Message': text_messages, 'Category': 'spam'}, ignore_index=True)
            sentences.append(text_messages)
            labels.append('spam')
            training_size = int(len(sentences) * 0.8)
            training_sentences = sentences[:training_size]
            training_labels = labels[:training_size]
            training_labels_final = np.array(training_labels)

            return

    # Tokenize and pad the input message
    sample_sequences = tokenizer.texts_to_sequences([text_messages])
    fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

    # Use the trained model to predict spam
    classes = model.predict(fakes_padded)

    # Check if the class is closer to 1 (indicating spam)
    if classes[0] >= 0.5:
        messagebox.showwarning("Spam Alert", "The given message is classified as SPAM.")
        # Add the new message to the dataset as spam
        dataset = dataset.append({'Message': text_messages, 'Category': 'spam'}, ignore_index=True)
        sentences.append(text_messages)
        labels.append('spam')
        training_size = int(len(sentences) * 0.8)
        training_sentences = sentences[:training_size]
        training_labels = labels[:training_size]
        training_labels_final = np.array(training_labels)

    else:
        print("The given message is classified as HAM (not spam).")

    # Tokenize and pad the input message
    sample_sequences = tokenizer.texts_to_sequences([text_messages])
    fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

    # Use the trained model to predict spam
    classes = model.predict(fakes_padded)

    # Check if the class is closer to 1 (indicating spam)
    if classes[0] >= 0.5:
        print("The given message is classified as SPAM.")

        # Add the new message to the dataset as spam
        dataset = dataset.append({'Message': text_messages, 'Category': 'spam'}, ignore_index=True)
        sentences.append(text_messages)
        labels.append('spam')
        training_size = int(len(sentences) * 0.8)
        training_sentences = sentences[:training_size]
        training_labels = labels[:training_size]
        training_labels_final = np.array(training_labels)

    else:
        print("The given message is classified as HAM (not spam).")


# Prompt the user to enter a message
text_messages = input("Enter Message: ")

# Predict whether the message is spam or not
predict_spam(text_messages)
