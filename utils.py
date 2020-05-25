import numpy as np
import os
import math
from skimage import io, transform
import matplotlib.pyplot as plt


def load_path(path):

    directories = []

    if os.path.isdir(path):
        directories.append(path)

    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            directories = directories + load_path(os.path.join(path, i))
            directories.append(os.path.join(path, i))

    return directories


def load_images_from_directories(dirs, ext, scale=1):
    
    # Load images from the given directory dirs (Resize the image if needed)

    images = []

    for dir in dirs:
        for f in os.listdir(dir):
            if f.endswith(ext):
                image = io.imread(os.path.join(dir, f))
                if scale > 1:
                    image = transform.resize(image, (image.shape[0] // scale, image.shape[1] // scale), anti_aliasing=True)
                    image = image * 255.
                if len(image.shape) > 2:
                    # 3 channels
                    images.append(image.astype(np.uint8))

    print(f'The dataset contains {len(images)} images')

    return np.array(images)


def load_data(directory, ext, scale=2):
    
    return load_images_from_directories(load_path(directory), ext, scale=scale)


def hr_images(images):
    
    return np.array(images)


def normalize_images(images):
    
    # Normalizing the images to range [-1, 1]

    return (images.astype(np.float16) - 127.5) / 127.5


def denormalize_images(images):
    
    # Denormalizing the images
    
    return ((images + 1) * 127.5).astype(np.uint8)


def lr_images(org_images, scale):
    
    # Resizing images to generate the input low-resolution images

    lr_images = []
    for org_image in org_images:
        lr_image = transform.resize(org_image, (org_image.shape[0] // scale, org_image.shape[1] // scale), anti_aliasing=True)
        lr_image = lr_image * 255
        lr_images.append(lr_image.astype(np.uint8))
    return np.array(lr_images)


def load_training_data_as_images(directory, ext, train_test_ratio=0.8):
    
    # load data from given directory
    images = load_data(directory, ext)
    num_training_data = int(images.shape[0] * train_test_ratio)

    print('--- Loading HR training images and creating LR training images')
    X_train_hr = hr_images(images[:num_training_data])
    X_train_lr = lr_images(X_train_hr, 4)

    print(f'>>> Number of training instances: {X_train_hr.shape[0]}')

    # normalizing train set
    X_train_hr = normalize_images(X_train_hr)
    X_train_lr = normalize_images(X_train_lr)

    print('--- Loading HR testing images and creating LR testing images')
    X_test_hr = hr_images(images[num_training_data:])
    X_test_lr = lr_images(X_test_hr, 4)

    # normalizing test set
    X_test_hr = normalize_images(X_test_hr)  
    X_test_lr = normalize_images(X_test_lr)

    print(f'>>> Number of testing instances: {X_test_hr.shape[0]}')

    return (X_train_lr, X_train_hr), (X_test_lr, X_test_hr)


def plot_generated_images(output_dir, epoch, generator, hr_images, lr_images):
    
    # Plot and save result images (original and generated high resolution images)
    
    num_images = lr_images.shape[0]
    image_size = hr_images.shape[1]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if epoch is None:
        FOLDER_NAME = 'final'
    else:
        FOLDER_NAME = f'epoch {epoch}'

    folder_path = os.path.join(output_dir, FOLDER_NAME)
    gen_path = os.path.join(folder_path, 'GEN')
    org_path = os.path.join(folder_path, 'ORG')

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    if not os.path.exists(org_path):
        os.mkdir(org_path)
        
    org_images = denormalize_images(hr_images)
    gen_images = generator.predict(lr_images)
    de_gen_images = denormalize_images(gen_images)

    for i in range(num_images):
        plt.imshow(de_gen_images[i])
        plt.axis('off')
        plt.savefig(os.path.join(gen_path, f'image_gen_{i}.png'))

    for i in range(num_images):
        plt.imshow(org_images[i])
        plt.axis('off')
        plt.savefig(os.path.join(org_path, f'image_org_{i}.png'))

    plt.close('all')