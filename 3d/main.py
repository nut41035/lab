import tensorflow as tf
import os
import datetime

from model import *
from data import *
from loss import *
from predict import *

from tensorflow.compat.v1 import ConfigProto, GPUOptions, InteractiveSession

def train_and_eval(EPOCHS = 20, BS = 16, IMAGE_COUNT = 139, VALIDATION_COUNT =134, learning_rate = 0.05, beta = 0.5):
    # session setting
    """ os.environ['TF_CPP_MIN_LOG_LEVEL']
      0 = all messages are logged (default behavior)
      1 = INFO messages are not printed
      2 = INFO and WARNING messages are not printed
      3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))

    ## program parameter
    BASE_DIR = './data/processed/3-channel-1/'
    TRAIN_DIR_PATH = BASE_DIR + 'train/'
    VALIDATION_DIR_PATH = BASE_DIR + 'validation/'
    seed = 1
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir="logs/fit/" + time_stamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer = tf.summary.create_file_writer(log_dir)

    ## training parameter
    loss_func = tversky_loss(beta)
    steps_per_epoch = 100
    input_size = (64,64,32,3)

    ## construct training and validation set
    training_data = DataGenerator(TRAIN_DIR_PATH, batch_size=BS, image_size=64)
    validating_data = DataGenerator(VALIDATION_DIR_PATH, batch_size=BS, image_size=64)

    ## load model
    model = unet_norm(input_size = input_size,loss_func=loss_func,l_rate=learning_rate)
    print('#### Model loaded')

    ## training begin
    model.fit_generator(training_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS,
                    validation_data=validating_data,
                    callbacks=[tensorboard_callback])

    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    model.save("model/UNet_%s.h5" %time_stamp)
    print("model saved at   model/UNet_%s.h5"%time_stamp)

    text = 'UNet_%s.h5\n\
            loss: weighted_dice  %s\n\
            learninf rate: %s\n\
            image size: %s\n'\
            %(time_stamp, beta,learning_rate,input_size)
    with open("./log.txt", "a") as myfile:
        myfile.write(text)

    ## prediction begin
    TP_sum, FP_sum, FN_sum = predict_folder(model, '%stest/'%BASE_DIR, save_mode=3, save_dir='./result/%s'%(time_stamp))
    eval_precicion = TP_sum/(TP_sum+FN_sum)
    eval_recall = TP_sum/(TP_sum+FP_sum)
    text = 'Evaluation result\n\
            TP : %s\n\
            FP : %s\n\
            FN : %s\n\
            Recall: %s\n\
            Precision: %s\n\n\n'\
            %(model_name, TP_sum, FP_sum, FN_sum, eval_recall, eval_precision)
    with open("./log.txt", "a") as myfile:
        myfile.write(text)

    InteractiveSession.close(session)

if __name__ == "__main__":
    train_and_eval()