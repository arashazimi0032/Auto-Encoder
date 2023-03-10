from mlxtend.data import loadlocal_mnist
from AutoEncoder import AutoEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

X, _ = loadlocal_mnist(images_path='./data/minist/t10k-images.idx3-ubyte',
                       labels_path='./data/minist/t10k-labels.idx1-ubyte')
X = X.astype(float) / 255
X[X > 0] = 1

X_train, X_test = train_test_split(X, test_size=0.3, random_state=101)

auto_encoder = AutoEncoder(encoder_dims=[64, 32, 16])

auto_encoder.build_model(input_shape=(len(X_train[0]), ))

auto_encoder.summary()

optimizer = Adam(learning_rate=0.001)

auto_encoder.compile(loss='mse', optimizer=optimizer)

auto_encoder.fit(X_train, X_train, epochs=200, batch_size=128, validation_data=(X_test, X_test), workers=8)

pred = auto_encoder.predict(X_test)

pd.DataFrame(auto_encoder.history.history).plot()

plt.figure(figsize=(12, 8))
for i in range(1, 33):
    plt.subplot(8, 8, i * 2 - 1)
    plt.imshow(pred[i].reshape(28, 28))
    plt.subplot(8, 8, i * 2)
    plt.imshow(X_test[i].reshape(28, 28))

plt.figure(figsize=(12, 8))
for i in range(1, 65):
    plt.subplot(8, 8, i)
    plt.imshow(auto_encoder.model.weights[-2][i-1, :].numpy().reshape(28, 28))

plt.show()
