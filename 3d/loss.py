from keras import backend as K
import tensorflow as tf
import random as r
def recall(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = K.abs(y_true)
    return ((numerator+1e-5) / (tf.reduce_sum(denominator, axis=-1)+1e-5))

def precision(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = K.abs(y_pred)
    return ((numerator+1e-5) / (tf.reduce_sum(denominator, axis=-1)+1e-5))


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred, smooth=1e-10):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """ 
    numerator = 2. * K.sum(K.abs(y_true * y_pred), axis=-1)
    denorminator = K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1)
    return (numerator + smooth) / (denorminator + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_dice_loss(beta):
    #more beta => focus on FP
    def loss(y_true, y_pred, smooth=1e-10):
        numerator = K.sum(K.abs(y_true * y_pred), axis=-1)
        denominator = K.sum(K.abs(y_true * y_pred), axis=-1) + K.sum(K.abs(beta * (1 - y_true) * y_pred), axis=-1) + K.sum(K.abs((1 - beta) * y_true * (1 - y_pred)), axis=-1)
        res = (numerator+smooth) / (denominator + smooth)
        return 1 - res
    return loss
def tversky_loss(beta):
    #more beta => focus on FN
    def tversky(y_true, y_pred, alpha=1-beta, beta=beta, smooth=1e-10):
        """ Tversky loss function.

        Parameters
        ----------
        y_true : keras tensor
            tensor containing target mask.
        y_pred : keras tensor
            tensor containing predicted mask.
        alpha : float
            real value, weight of '0' class.
        beta : float
            real value, weight of '1' class.
        smooth : float
            small real value used for avoiding division by zero error.

        Returns
        -------
        keras tensor
            tensor containing tversky loss.
        """
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return 1-answer
    return tversky
def f_score(tp,fp,fn,b):
    #recall is b times as important as precision
    try:
        p = tp/(tp+fp)
    except ZeroDivisionError:
        p = 0

    try:
        r = tp/(tp+fn)
    except ZeroDivisionError:
        r = 0
    
    try:
        x = (1+b*b)*((p*r)/(b*b*p+r))
    except ZeroDivisionError:
        x = 0
    return x

def balanced_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.compat.v1.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss * (1 - beta))

  return loss