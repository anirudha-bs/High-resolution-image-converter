from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_batches(just_load_dataset=False):

    batches = 256  
    batch = 0  
    batch_nb = 0  
    max_batches = -1  
    ep = 4  

    images = []
    x_train_n = []
    x_train_down = []

    x_train_n2 = []  
    x_train_down2 = [] 

    for root, dirnames, filenames in os.walk("DATA_DIRECTORY"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                if batch_nb == max_batches:  
                    return x_train_n2, x_train_down2
                filepath = os.path.join(root, filename)
                image = pyplot.imread(filepath)
                if len(image.shape) > 2:

                    image_resized = resize(image, (256, 256))   
                    x_train_n.append(image_resized)  
                    x_train_down.append(rescale(rescale(image_resized, 0.5), 2.0))  
                    batch += 1
                    if batch == batches:
                        batch_nb += 1

                        x_train_n2 = np.array(x_train_n)
                        x_train_down2 = np.array(x_train_down)

                        if just_load_dataset:
                            return x_train_n2, x_train_down2

                        print('Training batch', batch_nb, '(', batches, ')')

                        autoencoder.fit(x_train_down2, x_train_n2,
                                        epochs=ep,
                                        batch_size=10,
                                        shuffle=True,
                                        validation_split=0.15)

                        x_train_n = []
                        x_train_down = []

                        batch = 0

    return x_train_n2, x_train_down2


input_img = Input(shape=(256, 256, 3))
l1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)

encoder = Model(input_img, l7)

encoder.summary()

l8 = UpSampling2D()(l7)
l9 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l9)
l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l13)
l15 = add([l14, l2])

decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

x_train_n, x_train_down = train_batches(just_load_dataset=True)

autoencoder.load_weights("sr.img_net.mse.final_model5.patch.weights.best.hdf5")

out = np.clip(autoencoder.predict(x_train_down), 0.0, 1.0)

image_index = np.random.randint(0,UPPER_LIMIT_OF_DATA)

plt.imshow(x_train_down[image_index])
plt.show()
plt.imshow(out[image_index])
plt.show()
plt.imshow(x_train_n[image_index])
plt.show()
