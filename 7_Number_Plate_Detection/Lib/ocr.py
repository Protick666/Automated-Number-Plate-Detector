import tensorflow as tf
import random
import itertools
import numpy as np
from keras import backend as K
from keras.callbacks import History
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
import cv2

sess = tf.Session()
K.set_session(sess)

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dhaka', 'metro', 'ka', 'kha', 'ga', 'gha', 'cho', 'so',
           'mo', 'na', 'bo',
           'chatto', 'ha', 'chandpur', 'khulna', 'jo']


def get_output_size():
    return len(letters) + 1


def labels_to_text(labels):
    return ' '.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


class DataSet:

    def __init__(self,
                 image_dir,
                 annotations,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=9):

        self.image_dir = image_dir
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        self.samples = []

        for annotation in annotations:
            img_filepath = annotation[0]
            description = annotation[1].split('-')
            self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            # print(img_filepath)
            img = cv2.imread(self.image_dir + '/' + img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class OCR_NN:
    def __init__(self, img_w, train_anno, validation_anno, train_image_dir, validation_image_dir, load):

        self.load = load
        self.validation_image_dir = validation_image_dir
        self.train_image_dir = train_image_dir
        self.validation_anno = validation_anno
        self.train_anno = train_anno
        self.img_w = img_w
        self.model = None
        self.img_h = 64
        self.max_text_len = 9

        self.conv_filters = 16
        self.kernel_size = (3, 3)
        self.pool_size = 2
        self.time_dense_size = 32
        self.rnn_size = 512
        self.batch_size = 32
        self.downsample_factor = self.pool_size ** 2
        self.input_shape = None
        self.y_pred = None
        self.input_data = None
        self.model_name = 'model2.h5'

        self.valid = DataSet(self.validation_image_dir, self.validation_anno, self.img_w, self.img_h, self.batch_size,
                             self.downsample_factor)
        self.ds = DataSet(self.train_image_dir, self.train_anno, self.img_w, self.img_h, self.batch_size,
                          self.downsample_factor)

        self.ds.build_data()
        self.valid.build_data()

    def setTrainingData(self, train_anno, train_img_dir):
        self.train_anno = train_anno
        self.train_image_dir = train_img_dir
        self.ds = DataSet(self.train_image_dir, self.train_anno, self.img_w, self.img_h, self.batch_size,
                          self.downsample_factor)
        self.ds.build_data()

    def setValidationData(self, valid_anno, valid_img_dir):
        self.validation_anno = valid_anno
        self.validation_image_dir = valid_img_dir
        self.valid = DataSet(self.validation_image_dir, self.validation_anno, self.img_w, self.img_h, self.batch_size,
                             self.downsample_factor)
        self.valid.build_data()

    def getTestDataSet(self, test_anno, test_image_dir):
        test_ds = DataSet(test_image_dir, test_anno, self.img_w, self.img_h, self.batch_size, self.downsample_factor)
        test_ds.build_data()
        return test_ds

    def setNetworkParam(self, param_dict):
        self.conv_filters = param_dict['conv_filters']
        self.kernel_size = (param_dict['kernel_size'], param_dict['kernel_size'])
        self.pool_size = param_dict['pool_size']
        self.time_dense_size = param_dict['time_dense_size']
        self.rnn_size = param_dict['rnn_size']
        self.batch_size = param_dict['batch_size']

    def build_model(self):

        print('building model...')

        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, self.img_w, self.img_h)
        else:
            self.input_shape = (self.img_w, self.img_h, 1)

        self.ds.build_data()

        self.valid.build_data()

        act = 'relu'

        self.input_data = Input(name='the_input', shape=self.input_shape, dtype='float32')
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(self.input_data)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max1')(inner)
        inner = Dropout(0.2, name='drop1')(inner)
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = Dropout(0.2, name='drop2')(inner)
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv3')(inner)
        inner = BatchNormalization()(inner)

        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max2')(inner)

        conv_to_rnn_dims = (
            self.img_w // (self.pool_size ** 2), (self.img_h // (self.pool_size ** 2)) * self.conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru2_b')(
            gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))

        self.y_pred = Activation('softmax', name='softmax')(inner)

        # Model(inputs=self.input_data, outputs=self.y_pred).summary()

        labels = Input(name='the_labels', shape=[self.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [self.y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        self.model = Model(inputs=[self.input_data, labels, input_length, label_length], outputs=loss_out)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: self.y_pred}, optimizer=sgd)

    def load_saved_model(self):
        self.model.load_weights(self.model_name)

    def train_model(self):

        train_history = self.model.fit_generator(generator=self.ds.next_batch(),
                                                 steps_per_epoch=self.ds.n ,
                                                 epochs=5,
                                                 validation_data=self.valid.next_batch(),
                                                 validation_steps=self.valid.n)

        validation_loss = train_history.history['val_loss']

        self.model.save_weights(self.model_name)

        return validation_loss

    def test_model(self, test_ds):
        test_loss = self.model.evaluate_generator(generator=test_ds.next_batch(),steps=test_ds.n)
        return test_loss


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c] + ' '
        ret.append(outstr)
    return ret
