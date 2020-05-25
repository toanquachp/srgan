import numpy as np
from utils import load_training_data_as_images, plot_generated_images
from model import build_adversarial_model


def train_gan(models, data, parameters):
    
    '''
    Train adversarial model
    Steps:
    1. Generate fake HR images and get real HR images 
    2. Train discriminator (label fake HR images = 0, real HR images = 1)
    3. Get real HR images, LR images
    4. Train adversarial model (label 1 for adversarial training)
    5. Save and plot images on interval
    '''

    generator, discriminator, adversarial = models
    (X_train_lr, X_train_hr), (X_test_lr, X_test_hr) = data
    batch_size, train_steps, model_name, save_interval = parameters

    train_size = X_train_lr.shape[0]
    print(f'Train size:{X_train_lr.shape[0]}')

    rand_ind = np.random.randint(0, X_test_hr.shape[0], size=16)
    test_lr_images_per_epoch = X_test_lr[rand_ind]
    test_hr_images_per_epoch = X_test_hr[rand_ind]

    for i in range(train_steps):
        
        # train discriminator

        # get random lr images
        rand_ind = np.random.randint(0, train_size, size=batch_size)
        rand_lr_images = X_train_lr[rand_ind]
        
        # correspoding real hr images
        batch_real_hr_images = X_train_hr[rand_ind]
        # fake hr images is taken by using generator
        batch_fake_hr_images = generator.predict(rand_lr_images)

        batch_X = np.concatenate((batch_fake_hr_images, batch_real_hr_images))

        # label fake (label = 0) and real images (label = 1)
        batch_y = np.zeros(shape=(batch_size * 2,))
        batch_y[batch_size:] = 1.

        loss, acc = discriminator.train_on_batch(batch_X, batch_y)
        log = f'Step {i}: [discriminator: loss={loss}, acc={acc}]'

        # train adversarial

        # get random lr images and corresponding real hr images
        rand_ind = np.random.randint(0, train_size, size=batch_size)
        batch_lr_images = X_train_lr[rand_ind]
        batch_hr_images = X_train_hr[rand_ind]

        # fake label (label = 1)
        batch_y_label = np.ones(shape=(batch_size,))
        batch_y = [batch_hr_images, batch_y_label]

        metrics = adversarial.train_on_batch(batch_lr_images, batch_y)
    
        log = f'{log} [adversarial: loss={metrics[0]}, acc={metrics[-1]}'

        print(log)

        if i % save_interval == 0:
            generator.save(f'{model_name}_{i}.h5')
            plot_generated_images('./output_images/', epoch=i, generator=generator, hr_images=test_hr_images_per_epoch, lr_images=test_lr_images_per_epoch)
    
    generator.save(f'{model_name}_final.h5')

    plot_generated_images('./output_images/final', epoch=None, generator=generator, hr_images=X_test_hr, lr_images=X_test_lr)

    return generator


if __name__ == '__main__':
    DATA_DIRECTORY = './danbooru'
    # BATCH_SIZE = 64
    BATCH_SIZE = 6
    # TRAIN_STEPS = 10000
    TRAIN_STEPS = 100
    # SAVE_INTERVAL = 500
    SAVE_INTERVAL = 20
    MODEL_NAME = 'srgan'

    parameters = (BATCH_SIZE, TRAIN_STEPS, MODEL_NAME, SAVE_INTERVAL)

    # load data for training and testing
    print('-- Loading data')
    train_set, test_set = load_training_data_as_images(DATA_DIRECTORY, 'jpg', train_test_ratio=0.95)

    # lr image shape and hr image shape
    lr_shape = (train_set[0].shape[1], train_set[0].shape[2], train_set[0].shape[3])
    hr_shape = (train_set[1].shape[1], train_set[1].shape[2], train_set[1].shape[3])

    print('-- Building models')
    models = build_adversarial_model(hr_shape=hr_shape, lr_shape=lr_shape)

    print('-- Training models')
    train_gan(models=models, data=(train_set, test_set), parameters=parameters)

    print('>> Finished training')

    