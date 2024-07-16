import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


# Load the dataset
dataset = pd.read_csv('spam.csv')
# Load the trained model
model = load_model('spam_detection_model.h5')
# Load the tokenizer
tokenizer = Tokenizer(num_words=600, oov_token="<OOV>")
tokenizer.fit_on_texts(dataset['Message'])

def predict_spam(text_messages, tokenizer):
    # Tokenize and pad the input message
    sample_sequences = tokenizer.texts_to_sequences([text_messages])
    fakes_padded = pad_sequences(sample_sequences, padding='post', maxlen=60)
    # Use the trained model to predict spam
    classes = model.predict(fakes_padded)
    # Check if the class is closer to 1 (indicating spam)
    if classes[0] >= 0.5:
        messagebox.showwarning("Spam Alert", "The given message is classified as SPAM.")
    else:
        messagebox.showinfo("Spam Alert", "The given message is classified as HAM (not spam).")

def check_spam(tokenizer):
    text_messages = entry.get()
    predict_spam(text_messages, tokenizer)

# Create GUI window
root = tk.Tk()
root.title("Spam Detector")

# Create label and entry for message input
label = tk.Label(root, text="Enter Message:")
label.pack()
entry = tk.Entry(root, width=50)
entry.pack()

# Create button to check spam
button = tk.Button(root, text="Check Spam", command=lambda: check_spam(tokenizer))
button.pack()

# Run the GUI
root.mainloop()
