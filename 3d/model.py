from tensorflow import keras
from layers import *
from loss import *
# from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Concatenate,BatchNormalization

def unet(pretrained_weights = None,input_size = (64,64,32,3),l_rate=0.01,loss_func=dice_coef_loss):
    inputs = keras.layers.Input(input_size)
    conv1, pool1 = down_sampling(inputs, 8)
    conv2, pool2 = down_sampling(pool1, 16)
    conv3, pool3 = down_sampling(pool2, 32)
    conv4 = keras.layers.Conv3D(64, 3, padding = "same", activation = 'relu')(pool3)
    conv4 = keras.layers.Conv3D(64, 3, padding = "same", activation = 'relu')(conv4)
    up1 = up_sampling(conv4, conv3, 32)
    up2 = up_sampling(up1, conv2, 16)
    up3 = up_sampling(up2, conv1, 8)

    outputs = keras.layers.Conv3D(1, 1, padding="same", activation="sigmoid")(up3)

    model = keras.models.Model(inputs, outputs)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate), loss = loss_func, metrics = ['accuracy'])
    # model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
