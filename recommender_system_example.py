import pandas as pd
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder
from Utils import convert_to_user_item, custom_mse_loss

training_set = pd.read_csv(r".\data\ml-100k\u1.base", delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv(r".\data\ml-100k\u1.test", delimiter='\t')
test_set = np.array(test_set, dtype='int')

n_user = np.max([np.max(training_set[:, 0]), np.max(test_set[:, 0])])
n_movies = np.max([np.max(training_set[:, 1]), np.max(test_set[:, 1])])

training_set = convert_to_user_item(training_set, n_user, n_movies)
test_set = convert_to_user_item(test_set, n_user, n_movies)

auto_encoder = AutoEncoder(encoder_dims=[64, 32, 16])

auto_encoder.build_model(input_shape=(n_movies, ), no_data_value=0)

auto_encoder.summary()

optimizer = Adam(learning_rate=0.001)

auto_encoder.compile(loss=custom_mse_loss, optimizer=optimizer)

auto_encoder.fit(training_set, training_set, epochs=100, batch_size=128, validation_data=(training_set, training_set),
                 workers=8)

pred = auto_encoder.predict(training_set)

pd.DataFrame(auto_encoder.history.history).plot()
plt.show()
