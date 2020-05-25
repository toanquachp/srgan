import tensorflow
tensorflow.config.experimental_run_functions_eagerly(True)

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import Lambda, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K


class VGG_LOSS(object):
    
    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):

        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        
        # Freeze all layer of vgg9
        for l in vgg19.layers:
            l.trainable = False
        
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        # return K.mean(K.square(y_true - y_pred))
        return K.mean(K.square(model(y_true) - model(y_pred)))


def build_residual_blocks(model, kernel_size=3, filters=64, strides=1):
    
    # Build a residual block for the Generator

    gen_model = model

    res_block = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(model)
    res_block = BatchNormalization(momentum = 0.5)(res_block)
    res_block = PReLU(shared_axes=[1, 2])(res_block)
    res_block = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(res_block)
    res_block = BatchNormalization(momentum = 0.5)(res_block)

    model = add([gen_model, res_block])

    return model


def build_sub_pixel_block(model, kernel_size=3, filters=256, strides=1):
      
    # Build the up-sampling block for the generator
    
    # TODO: test conv2dtranspose
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(model)
    model = UpSampling2D(size=2)(model)
    model = PReLU(shared_axes=[1, 2])(model)

    return model


def build_generator(lr_shape):
    
    # Build the generator

    gen_input = Input(shape=lr_shape)

    gen = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(gen_input)
    gen = PReLU(shared_axes=[1, 2])(gen)

    res = gen

    for _ in range(6):
        gen = build_residual_blocks(gen, 3, 64, 1)

    gen = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(gen)
    gen = BatchNormalization(momentum=0.5)(gen)
    gen = add([res, gen])

    for _ in range(2):
       gen = build_sub_pixel_block(gen, 3, 256, 1)

    gen = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen)
    gen = Activation('tanh')(gen)

    generator_model = Model(inputs=gen_input, outputs=gen, name='generator')

    plot_model(generator_model, to_file='models/images/generator.png')

    return generator_model


def build_conv_block(model, kernel_size=3, filters=64, strides=1):
      
    # Build convolutional block for discriminator

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def build_discriminator(hr_shape):
        
    # Build the discriminator
    
    kernel_size = 3
    filters = [64, 128, 128, 256, 256, 512, 512]

    dis_input = Input(shape=hr_shape)
    dis = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(dis_input)
    dis = LeakyReLU(alpha=0.2)(dis)

    # make loop
    dis = build_conv_block(dis, kernel_size=3, filters=64, strides=2)
    dis = build_conv_block(dis, kernel_size=3, filters=128, strides=1)
    dis = build_conv_block(dis, kernel_size=3, filters=128, strides=2)
    dis = build_conv_block(dis, kernel_size=3, filters=256, strides=1)
    dis = build_conv_block(dis, kernel_size=3, filters=256, strides=2)
    dis = build_conv_block(dis, kernel_size=3, filters=512, strides=1)
    dis = build_conv_block(dis, kernel_size=3, filters=512, strides=2)
    
    dis = Flatten()(dis)
    # dis = Dense(1024)(dis)
    # dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dense(1)(dis)
    dis = Activation('sigmoid')(dis)

    discriminator_model = Model(inputs=dis_input, outputs=dis, name='discriminator')

    plot_model(discriminator_model, to_file='models/images/discriminator.png')

    return discriminator_model


def get_optimizer():
    
    # Get optimizer for training SRGAN

    return Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def build_adversarial_model(hr_shape=(256, 256, 3), lr_shape=(64, 64, 3)):
      
    # Build adversarial model
    
    # vgg_loss and optimizer
    optimizer = get_optimizer()
    vgg_loss = VGG_LOSS(hr_shape)

    # build discriminator model
    discriminator = build_discriminator(hr_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    discriminator.summary()
    
    # build generator model
    generator = build_generator(lr_shape)
    generator.compile(loss=vgg_loss.vgg_loss, optimizer=optimizer)
    generator.summary()

    # build gan model
    discriminator.trainable = False
    
    gan_input = Input(shape=lr_shape)
    generator_output = generator(gan_input)
    discriminator_output = discriminator(generator_output)
    
    gan = Model(inputs=gan_input, outputs=[generator_output, discriminator_output], name='adversarial_model')

    plot_model(gan, to_file='models/images/gan.png')

    gan.compile(loss=[vgg_loss.vgg_loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer=optimizer, metrics=['accuracy'])

    # TODO: test mse loss function
    # gan.compile(loss=[vgg_loss.vgg_loss, 'mse'], loss_weights=[1., 1e-3], optimizer=optimizer, metrics=['accuracy'])

    gan.summary()
    
    models = (generator, discriminator, gan)
    
    return models

