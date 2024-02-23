# Day 15: Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are a class of artificial intelligence models used for generative tasks. They are composed of two main components: a generator and a discriminator. The generator creates new data instances, while the discriminator evaluates them for authenticity; i.e., whether each instance of data that it reviews belongs to the actual training dataset or not.

## Task 1: Components of GANs

### Generator

The generator in a GAN is a neural network. Its role is to generate new data. It takes random noise as input and returns data in the desired format. The generator does not know the real data distribution, and its goal is to fool the discriminator into believing the data it generated is real.

### Discriminator

The discriminator in a GAN is also a neural network. It takes both real data instances from the training set and fake instances from the generator as input, and it returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.

## Task 2: Implement a Basic GAN Model

Here is a simple implementation of a GAN model using Python and Keras:

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Generator
def create_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(784, activation='tanh'))
    return model

# Discriminator
def create_discriminator():
    model = Sequential()
    model.add(Dense(1024, input_dim=784, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

# GAN
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())
    return gan
```

## Task 3: Train the GAN Model

Training a GAN involves running a loop over the number of epochs, within which we run another loop over the number of batches. For each batch, we perform the following steps:

1. **Train the discriminator**: We first generate a batch of fake images using the generator. We then create a batch of real images from the training set. We train the discriminator on these two batches, one real and one fake.

2. **Train the generator**: We generate a batch of fake images. These are fed into the discriminator with labels of real images. If the generator is performing well, the discriminator will classify these fake images as real.

The quality of the generated images can be evaluated visually, or more objectively using metrics such as the Inception Score or the Frechet Inception Distance.

```python
def train_gan(gan, generator, discriminator, epochs=100, batch_size=128):
    for e in range(epochs):
        for _ in range(batch_size):
            noise = np.random.normal(0, 1, [batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
```

This concludes our tutorial on Generative Adversarial Networks. We have learned about the components of GANs, implemented a basic GAN model, and trained it to generate new images.