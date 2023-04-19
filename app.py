import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras as keras


import tensorflow_wavelets.Layers.DWT as DWT
from keras.models import load_model
import nibabel as nib
import numpy as np
import plotly.graph_objs as go
import os
import matplotlib.pyplot as plt
from distutils.log import debug
from fileinput import filename
from flask import *
import tensorflow_addons as tfa
import plotly
import scipy
# import tensorflow as tf
# from sklearn.preprocessing import normalize
import cv2

from distutils.log import debug
from fileinput import filename
from flask import *
app = Flask(__name__, template_folder='tests', static_folder='src')


@app.route('/')
def main():
    return render_template("main.html")

# returns a compiled model
# identical to the previous one

# Pooling layer
#from Mish import Mish


def Dice(P, T):
    print(P, T)
    #print(logits, y)
    #o = tf.ones((256,256,1))
    #P = tf.where(P, o * 1.0, o * 0.0)
    I = tf.math.abs(tf.reduce_sum(tf.math.multiply(P, T)))
    U = tf.math.abs(tf.reduce_sum(T+P-(T*P)))
    # tf.math.subtract(tf.math.add(logits,y),tf.math.multiply(logits,y)))
    # tf.math.subtract(tf.constant(1.0, dtype=tf.float32),tf.math.divide(inter,union))
    dice = (2*I)/(U+I)
    return 1-dice


def IoU(P, T):
    #print(logits, y)
    #o = tf.ones((256,192,1))
    #P = tf.where(P, o * 1.0, o * 0.0)
    I = tf.math.abs(tf.reduce_sum(tf.math.multiply(P, T)))
    U = tf.math.abs(tf.reduce_sum(T+P-(T*P)))
    # tf.math.subtract(tf.math.add(logits,y),tf.math.multiply(logits,y)))
    # tf.math.subtract(tf.constant(1.0, dtype=tf.float32),tf.math.divide(inter,union))
    iou = (I+1)/(U+1)
    return (1-iou)/(1+iou)


# Model configuration options
# of trainable filter configs. for DWTUnet
config = {
    'config2': (2, 2, 2, 2, 2, 2),
    'config4': (4, 4, 4, 4, 4, 4),
    'config8': (8, 8, 8, 8, 8, 8),
    'config16': (16, 16, 16, 16, 16, 16),
    'config32': (32, 32, 32, 32, 32, 32),
    'config64': (64, 64, 64, 64, 64, 64),
    'config42': (4, 2, 2, 2, 2, 2),
    'config82': (8, 2, 2, 2, 2, 2),
    'config162': (16, 2, 2, 2, 2, 2),
    'config322': (32, 2, 2, 2, 2, 2),
    'config642': (64, 2, 2, 2, 2, 2),
    'config84': (8, 4, 4, 4, 4, 4),
    'config164': (16, 4, 4, 4, 4, 4),
    'config322': (32, 4, 4, 4, 4, 4),
    'config644': (64, 4, 4, 4, 4, 4),
    'config168': (16, 8, 8, 8, 8, 8),
    'config328': (32, 8, 8, 8, 8, 8),
    'config648': (64, 8, 8, 8, 8, 8),
    'config3216': (32, 16, 16, 16, 16, 16),
    'config6416': (64, 16, 16, 16, 16, 16),
    'config6432': (64, 32, 32, 32, 32, 32),
    'config132': (1, 2, 4, 8, 16, 32),
    'config264': (2, 4, 8, 16, 32, 64),
    'config4128': (4, 8, 16, 32, 64, 128),
    'config8256': (8, 16, 32, 64, 128, 256),
}

# CONFIG: Trainable filters
f, a, b, c, d, e = config['config2']
global wave
wave = 'haar'

activation = 'relu'
#activation = tfa.activations.mish

# activation2='sigmoid'
activation2 = 'relu'

activation3 = 'sigmoid'
# activation3='softmax' #bad perf.
activation3 = tfa.activations.mish


# BEST CONFIG. ***
activation, activation2, activation3 = 'relu', 'relu', tfa.activations.mish


class Pooling(tf.keras.layers.Layer):
    '''DWT Pooling Layer: keep Low freq band only
          #separableConv2D
          #Mish'''

    def __init__(self, Ψ='', **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.Ψ = 'haar'
        self.dwt = DWT.DWT(name=self.Ψ, concat=0)

    def call(self, inputs):
        """inputs -> wave0 -> wave1 #
        !!-> wave2 -> wave0_cap(inverse)"""
        chans = inputs.shape[3]
        wave0 = inputs  # L0
        wave1 = self.dwt(wave0)  # L1

        #wave2 = WaveTFFactory.build(wave_kern)(wave1[:,:,:,:chans])
        #inv_wave0 = WaveTFFactory().build(wave_kern, inverse = True)(wave2)

        #pool = DWT.DWT(name=wave,concat=0)(inputs)
        return wave1[:, :, :, :chans]
        # return tf.keras.layers.MaxPooling2D((2, 2))(inputs)
        # return tf.keras.layers.AveragePooling2D((2, 2))(inputs)


#import keras

# DWT
print(Pooling())


class dwt(tf.keras.layers.Layer):
    """DWT Layer"""

    def __init__(self):
        super(dwt, self).__init__()
        global wave
        self.dwt = DWT.DWT(name=wave, concat=0)

    def call(self, x):
        return self.dwt(x)

# IDWT


class idwt(tf.keras.layers.Layer):
    """IDWT Layer"""

    def __init__(self):
        super(idwt, self).__init__()
        global wave
        self.idwt = DWT.IDWT(wavelet_name=wave, concat=0)

    def call(self, x):
        return self.idwt(x)


class UpdatedMeanIoU(tf.keras.metrics.IoU):
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None, **kwargs):
        super(UpdatedMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


# Model configuration options
# of trainable filter configs. for DWTUnet
config2 = {
    'config2': (2, 2, 2, 2, 2, 2),
    'config4': (4, 4, 4, 4, 4, 4),
    'config8': (8, 8, 8, 8, 8, 8),
    'config16': (16, 16, 16, 16, 16, 16),
    'config32': (32, 32, 32, 32, 32, 32),
    'config64': (64, 64, 64, 64, 64, 64),
    'config42': (4, 2, 2, 2, 2, 2),
    'config82': (8, 2, 2, 2, 2, 2),
    'config162': (16, 2, 2, 2, 2, 2),
    'config322': (32, 2, 2, 2, 2, 2),
    'config642': (64, 2, 2, 2, 2, 2),
    'config84': (8, 4, 4, 4, 4, 4),
    'config164': (16, 4, 4, 4, 4, 4),
    'config322': (32, 4, 4, 4, 4, 4),
    'config644': (64, 4, 4, 4, 4, 4),
    'config168': (16, 8, 8, 8, 8, 8),
    'config328': (32, 8, 8, 8, 8, 8),
    'config648': (64, 8, 8, 8, 8, 8),
    'config3216': (32, 16, 16, 16, 16, 16),
    'config6416': (64, 16, 16, 16, 16, 16),
    'config6432': (64, 32, 32, 32, 32, 32),
    'config132': (1, 2, 4, 8, 16, 32),
    'config264': (2, 4, 8, 16, 32, 64),
    'config4128': (4, 8, 16, 32, 64, 128),
    'config8256': (8, 16, 32, 64, 128, 256),
}

# CONFIG: Trainable filters
f, a, b, c, d, e = config2['config16']

# Pooling layer
#from Mish import Mish


class Pooling2(tf.keras.layers.Layer):
    '''DWT Pooling Layer: keep Low freq band only
          #separableConv2D
          #Mish'''

    def __init__(self, Ψ='', **kwargs):
        super(Pooling2, self).__init__(**kwargs)
        self.supports_masking = True
        self.Ψ = 'haar'
        self.dwt2 = DWT.DWT(name=self.Ψ, concat=0)

    def call(self, inputs):
        """inputs -> wave0 -> wave1 #
        !!-> wave2 -> wave0_cap(inverse)"""
        chans = inputs.shape[3]
        wave0 = inputs  # L0
        wave1 = self.dwt2(wave0)  # L1

        #wave2 = WaveTFFactory.build(wave_kern)(wave1[:,:,:,:chans])
        #inv_wave0 = WaveTFFactory().build(wave_kern, inverse = True)(wave2)

        #pool = DWT.DWT(name=wave,concat=0)(inputs)
        return wave1[:, :, :, :1]
        # return tf.keras.layers.MaxPooling2D((2, 2))(inputs)
        # return tf.keras.layers.AveragePooling2D((2, 2))(inputs)


# DWT IDWT layers
#import keras


# DWT
class dwt2(tf.keras.layers.Layer):
    """DWT Layer"""

    def __init__(self, **args):
        super(dwt2, self).__init__()
        global wave
        self.dwt2 = DWT.DWT(wavelet_name=wave, concat=0)

    def call(self, x):
        return self.dwt2(x)


# IDWT
class idwt2(tf.keras.layers.Layer):
    """IDWT Layer"""

    def __init__(self, **args):
        super(idwt2, self).__init__()
        global wave
        self.idwt2 = DWT.IDWT(wavelet_name=wave, concat=0)

    def call(self, x):
        return self.idwt2(x)


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        test_load = np.array(nib.load(f.filename).get_fdata())

        # # FRONT VIEW
        # view1_front = scipy.ndimage.interpolation.rotate(
        #     test_load, angle=90, mode='nearest', axes=(1, 2))

        # # SIDE VIEW
        # view2_side = scipy.ndimage.interpolation.rotate(
        #     test_load, angle=270, mode='nearest', axes=(0, 2))
        # # print(np.max(test_load))
        # # test = nib.load("anita_4.nii")
        # # test_load.header.set_data_dtype(np.float64)

        # # print(test.get_data_dtype())
        # # hdr=test.header
        # # model.predict(test_load)
        # # print(test_load.shape)

        def frontal_view_imgs(data):
            list_frontal_view_imgs = []
            for i in range(data.shape[0]):
                im_slice = data[i, :, :]
                # im_resized = im_slice
                im_resized = cv2.resize(im_slice, (256, 256))
                del im_slice
                list_frontal_view_imgs.append(np.expand_dims(im_resized, -1))
            return np.array(list_frontal_view_imgs)

        front_view = frontal_view_imgs(test_load)
        new_model = load_model('final.h5', custom_objects={
            'Pooling': Pooling, 'dwt': dwt, 'idwt': idwt, })
        print("hi")

        pred = np.array(new_model(front_view))
        print(np.max(pred))
        converted_array = np.array(pred[:, :, :, 0])
        final_image = np.multiply(front_view[:, :, :, 0], converted_array)

        # # FRONT VIEW
        # view1 = scipy.ndimage.interpolation.rotate(
        #     final_image, angle=90, mode='nearest', axes=(1, 2))

        # # SIDE VIEW
        # view2 = scipy.ndimage.interpolation.rotate(
        #     final_image, angle=270, mode='nearest', axes=(0, 2))

        # normalized_vector = final_image / np.max(final_image)
        # normalized_vector1 = view1 / np.max(view1)
        # normalized_vector2 = view2 / np.max(view2)

        # normalized_vector_og = test_load / np.max(test_load)
        # normalized_vector1_og = view1_front / np.max(view1_front)
        # normalized_vector2_og = view2_side / np.max(view2_side)

        # view_top = nib.Nifti1Image(normalized_vector*255, affine=None)

        # nib.save(view_top, os.path.join('static', 'top.nii'))

        # view_front = nib.Nifti1Image(normalized_vector1*255, affine=None)

        # nib.save(view_front, os.path.join('static', 'front.nii'))
        # view_side = nib.Nifti1Image(normalized_vector2*255, affine=None)

        # nib.save(view_side, os.path.join('static', 'side.nii'))

        # view_top_og = nib.Nifti1Image(normalized_vector_og*255, affine=None)

        # nib.save(view_top_og, os.path.join('static', 'topview.nii'))

        # view_front_og = nib.Nifti1Image(normalized_vector1_og*255, affine=None)

        # nib.save(view_front_og, os.path.join('static', 'frontview.nii'))
        view_side_og = nib.Nifti1Image(final_image, affine=None)

        nib.save(view_side_og, os.path.join('src/data', 'image2.nii'))
        return render_template('debug_server.html')


@app.route('/mask', methods=['POST'])
def mask():
    test_load = np.array(nib.load(os.path.join(
        'src/data', 'image2.nii')).get_fdata())

    def frontal_view_imgs(data):
        list_frontal_view_imgs = []
        for i in range(data.shape[0]):
            im_slice = data[i, :, :]
            # im_resized = im_slice
            im_resized = cv2.resize(im_slice, (256, 256))
            del im_slice
            list_frontal_view_imgs.append(np.expand_dims(im_resized, -1))
        return np.array(list_frontal_view_imgs)

    front_view = frontal_view_imgs(test_load)
    new_model = load_model('best_seg_fin.h5', custom_objects={
        'Pooling': Pooling2, 'dwt': dwt2, 'idwt': idwt2, },)
    print("hi")
    pred = np.array(new_model(front_view))
    print(np.max(pred))
    final_image = np.argmax(pred, -1)

    # final_image = np.array(final_image[:, :, :, 0])
    # final_image = np.multiply(front_view[:, :, :, 0], converted_array)

    view_side_og = nib.Nifti1Image(final_image.astype(float), affine=None)

    nib.save(view_side_og, os.path.join('src/data', 'image2.nii'))

    return render_template('debug_server.html')


if __name__ == '__main__':
    app.run(debug=True)
