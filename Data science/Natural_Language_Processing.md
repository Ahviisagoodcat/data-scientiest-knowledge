# Natural Language Processing

- environment: 
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

#eg. sentences=['I love my dog','I love my cat'] ingnore sysmbol

tokenizer=Tokenizer(num_words=100) #set a limit of length ->common words
tokenizer.fit_on_tests(sentences)  #give each word a number
word_index=tokenizer.word_index #dictonary key:word, value:token

sequences=tokenizer.texts_to_sequences(sentences) #one hot encoding using the dictionary
```
