# Classifica-o-de-Texto-
Script : Classificação de Texto com TensorFlow e Keras

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dados de exemplo
textos = ["Este é um exemplo de texto.",
          "Outro exemplo de texto para classificação.",
          "Mais um exemplo para teste."]

# Tokenização dos textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(textos)
sequencias = tokenizer.texts_to_sequences(textos)

# Padronização das sequências
sequencias_padronizadas = pad_sequences(sequencias, padding='post')

# Definição do modelo de classificação
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
modelo.fit(sequencias_padronizadas, [0, 1, 1], epochs=10)
