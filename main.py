import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K



class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP(object):
    def __init__(self, img_shape=(28, 28, 1),  dims=100):
        self.img_shape = img_shape
        self.channels = img_shape[2]
        self.dims = dims
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and discriminator
        self.generator = self._gan_generator()
        self.generator.summary()
        self.discriminator = self._gen_discriminator()
        self.discriminator.summary()

        #-------------------------------
        # Construct Computational Graph
        #       for the discriminator
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'   # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc],
                                         outputs=[valid, fake, validity_interpolated])
        self.discriminator_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                                         optimizer=optimizer,
                                         loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.discriminator(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def _gan_generator(self):
        inputs = Input(shape=(100,))

        x = Dense(7 * 7 * self.dims * 4, activation='relu', input_dim=self.latent_dim)(inputs)
        x = Reshape((7, 7, self.dims * 4))(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.dims * 4, kernel_size=5, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.dims * 2, kernel_size=5, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        # x = UpSampling2D()(x)
        # x = Conv2D(self.dims * 1, kernel_size=5, padding="same")(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Activation("relu")(x)

        x = Conv2D(self.channels, kernel_size=5, padding="same")(x)
        x = Activation("tanh")(x)

        return Model(inputs, x)

    def _gen_discriminator(self):
        inputs = Input(shape=self.img_shape)

        x = Conv2D(self.dims, kernel_size=5, strides=2, padding="same", input_shape=self.img_shape)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(self.dims * 2, kernel_size=5, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        return Model(inputs, x)

        # c_x = Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
        # c_x = BatchNormalization(momentum=0.8)(c_x)
        # c_x = LeakyReLU(alpha=0.2)(c_x)
        # c_x = Dropout(0.25)(c_x)
        # c_x = Flatten()(c_x)
        # c_x = Dense(1, name='c')(c_x)
        #
        # d_x = Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
        # d_x = BatchNormalization(momentum=0.8)(d_x)
        # d_x = LeakyReLU(alpha=0.2)(d_x)
        # d_x = Dropout(0.25)(d_x)
        # s = d_x.get_shape().as_list()
        # d_x = Reshape()(d_x, [s[1] * s[2] * s[3]])(d_x)
        # d_x = Dense(1, name='d')(d_x)
        #
        # return Model(inputs, [d_x, c_x])

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.discriminator_model.train_on_batch([imgs, noise],
                                                                 [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)

