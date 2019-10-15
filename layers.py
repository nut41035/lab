from tensorflow import keras

def down_sampling(Input, target_size, kernel_size = 3, padding = "same", activation = "relu", stride = 1):
    conv = keras.layers.Conv2D(target_size, kernel_size, padding = padding, strides = stride)(Input)
    conv = keras.layers.Conv2D(target_size, kernel_size, padding = padding, strides = stride)(conv)
    pool = keras.layers.MaxPool2D(pool_size=(2, 2))(conv)
    return conv, pool

def up_sampling(Input1, Input2, target_size, kernel_size = 3, padding = "same", activation = "relu", stride = 1):
    # Input1 is from previous layer
    # Input2 is layer to concat
    pool = keras.layers.UpSampling2D((2, 2))(Input1)
    merge = keras.layers.Concatenate()([pool,Input2])
    conv = keras.layers.Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation=activation)(merge)
    conv = keras.layers.Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation=activation)(conv)
    return conv