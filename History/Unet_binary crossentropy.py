from tensorflow import keras
from layers import *

def unet(pretrained_weights = None,input_size = (64,64,3)):
    inputs = keras.layers.Input(input_size)
    conv1, pool1 = down_sampling(inputs, 64)
    conv2, pool2 = down_sampling(pool1, 128)
    conv3, pool3 = down_sampling(pool2, 256)
    conv4 = keras.layers.Conv2D(512, 3, padding = "same", activation = 'relu')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, padding = "same", activation = 'relu')(conv4)
    up1 = up_sampling(conv4, conv3, 256)
    up2 = up_sampling(up1, conv2, 128)
    up3 = up_sampling(up2, conv1, 64)

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up3)

    model = keras.models.Model(inputs, outputs)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.summary()


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model