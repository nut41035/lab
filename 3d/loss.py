from keras import backend as K
import tensorflow as tf
import random as r

def jaccard_distance_loss():
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
    return jaccard_distance

def dice_coef_loss():
    def dice_coef(y_true, y_pred, smooth=1e-10):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
            =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """ 
        numerator = 2. * K.sum(K.abs(y_true * y_pred), axis=-1)
        denorminator = K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1)
        return 1-((numerator + smooth) / (denorminator + smooth))
    return dice_coef

def tversky_loss(beta): ## orr weighted dice loss
    #more beta => focus on FN
    def tversky(y_true, y_pred, alpha=1-beta, beta=beta, smooth=1e-10):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return 1-answer
    return tversky
