# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open("./sarcasm.json","r") as f:
        data_store=json.load(f)
    for i in data_store:
        sentences.append(i["headline"])
        labels.append(i["is_sarcastic"])

    training_sentences=sentences[:training_size]
    validation_sentences=sentences[training_size:]

    training_labels=labels[:training_size]
    validation_labels=labels[training_size:]

    training_labels_fin=np.array(training_labels)
    validation_labels_fin=np.array(validation_labels)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok) # YOUR CODE HERE
    tokenizer.fit_on_texts(training_sentences)

    training_sequences=tokenizer.texts_to_sequences(training_sentences)
    training_padded=pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

    validation_sequences=tokenizer.texts_to_sequences(validation_sentences)
    validation_padded=pad_sequences(validation_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)


    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Conv1D(32,kernel_size=3,activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(12,activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(training_padded,training_labels_fin,epochs=30,validation_data=(validation_padded,validation_labels_fin))
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
