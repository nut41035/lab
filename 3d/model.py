from tensorflow import keras
from layers import *
from loss import *
# from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Concatenate,BatchNormalization

def unet(pretrained_weights = None,input_size = (64,64,32,3),l_rate=0.01,loss_func=dice_coef_loss):
    inputs = keras.layers.Input(input_size)
    conv1, pool1 = down_sampling_with_norm(inputs, 8)
    conv2, pool2 = down_sampling_with_norm(pool1, 16)
    conv3, pool3 = down_sampling_with_norm(pool2, 32)
    conv4, pool4 = down_sampling_with_norm(pool3, 64)
    conv5 = keras.layers.Conv3D(128, 3, padding = "same", activation = 'relu')(pool4)
    conv5 = keras.layers.Conv3D(128, 3, padding = "same", activation = 'relu')(conv5)
    up1 = up_sampling_with_norm(conv5, conv4, 64)
    up2 = up_sampling_with_norm(up1, conv3, 32)
    up3 = up_sampling_with_norm(up2, conv2, 16)
    up4 = up_sampling_with_norm(up3, conv1, 8)

    outputs = keras.layers.Conv3D(1, 1, padding="same", activation="sigmoid")(up4)

    model = keras.models.Model(inputs, outputs)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate), loss = loss_func, metrics =['accuracy'])
    # model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
