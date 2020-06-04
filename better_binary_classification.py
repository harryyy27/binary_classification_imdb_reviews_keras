from keras.datasets import imdb
from keras import models
from keras import layers
from parasite import parasite
import numpy as np
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict((key,value) for (value,key) in word_index.items())
# print(word_index.items())
#get method returns dictionary with key input
split_review = parasite.split(' ')
print(split_review)
encoded_review = list(filter(lambda x: x<10000, [list(reverse_word_index.keys())[list(reverse_word_index.values()).index(word)] for word in split_review  if word in list(reverse_word_index.values())]))
print(encoded_review)
for i in encoded_review:
    print(i)
    if len(str(i)) > 4 and i>9999:
        encoded_review.remove(i)
print(encoded_review)
np_array = np.asarray([encoded_review])
print(np_array)
print(train_data[0])

def vectorize_sequences(sequences, dimension=10000):
    #create zeros np array of shape (sequences length, dimension)
    results = np.zeros((len(sequences), dimension))
    print(sequences)
    print(enumerate(sequences))
    for i, sequence in enumerate(sequences):
    
        results[i,sequence]=1
    return results

x_train = vectorize_sequences(train_data)
x_test= vectorize_sequences(test_data)
p_site=vectorize_sequences(np_array)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mse',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)

print(model.predict(p_site))