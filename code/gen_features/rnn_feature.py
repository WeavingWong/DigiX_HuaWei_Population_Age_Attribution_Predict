import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, PReLU,ReLU
from keras.layers import Bidirectional, SpatialDropout1D, CuDNNGRU,CuDNNLSTM, Conv1D,Conv2D,MaxPool2D,Reshape
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D, concatenate,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
from keras.engine import Layer
from keras.layers.core import Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import  initializers, regularizers, constraints
from keras.layers import K, Activation

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
def RnnVersion1( n_recurrent=50, n_filters=30, dropout_rate=0.2, l2_penalty=0.0001,n_capsule = 10, n_routings = 5, capsule_dim = 16):
    K.clear_session()
    def conv_block(x, n, kernel_size):
        x = Conv1D(n, kernel_size, activation='relu') (x)
        x = Conv1D(n_filters, kernel_size, activation='relu') (x)
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAveragePooling1D()(x)
        x_max = GlobalMaxPooling1D()(x)
        return concatenate([x_att, x_avg, x_max])  
    def att_max_avg_pooling(x):
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAveragePooling1D()(x)
        x_max = GlobalMaxPooling1D()(x)
        return concatenate([x_att, x_avg, x_max])

    inputs = Input(shape=(100,))
    emb = Embedding(9399, 300,  trainable=True)(inputs)

    # model 0
    x0 = BatchNormalization()(emb)
    x0 = SpatialDropout1D(dropout_rate)(x0)
    
    x0 = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x0)
    x0 = Conv1D(n_filters, kernel_size=3)(x0)
    x0 = PReLU()(x0)
#     x0 = Dropout(dropout_rate)(x0)
    x0 = att_max_avg_pooling(x0)

    # model 1
    x1 = SpatialDropout1D(dropout_rate)(emb)
    x1 = Bidirectional(
        CuDNNGRU(2*n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x1)
    x1 = Conv1D(2*n_filters, kernel_size=2)(x1)
    x1 = PReLU()(x1)
#     x1 = Dropout(dropout_rate)(x1)
    x1 = att_max_avg_pooling(x1)

    x = concatenate([x0, x1],name='concatenate')
    
    fc = Dense(128, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(fc)#   , kernel_regularizer=l2(l2_penalty), activity_regularizer=l2(l2_penalty)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Nadam',metrics =['accuracy'])
    return model

user_app_actived = pd.read_csv('../../data/original_data/user_app_actived.csv',names=['uId', 'appId'])
train = pd.read_csv('../../data/original_data/age_train.csv',names=['uId','age_group'],dtype={'uId':np.int32, 'age_group':np.int8})
train = pd.merge(train, user_app_actived, how='left', on='uId')
y_train = train.age_group - 1
test = pd.read_csv('../../data/original_data/age_test.csv',names=['uId'],dtype={'uId':np.int32})
test = pd.merge(test, user_app_actived, how='left', on='uId')
 
x_train = train.appId.str.split('#')
x_test = test.appId.str.split('#')
#将appid映射为数字id
appId = pd.read_csv('../../data/processed_data/appId.csv')
app_dict = dict(zip(appId.appId,appId.id))
x_train = [[app_dict[x] for x in apps if  x in app_dict] for apps in x_train]
x_test = [[app_dict[x] for x in apps if  x in app_dict] for apps in x_test]
#序列长度填补为100
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


y_train = to_categorical(y_train, num_classes=None)

filepath="weights_best2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
callbacks = [checkpoint, reduce_lr]
model = RnnVersion1()

model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=1, callbacks=callbacks, 
         )
model.load_weights(filepath)


y_val = model.predict(x_train, batch_size=128, verbose=1)
y_test =  np.array(model.predict(x_test, batch_size=128, verbose=1))
    
y_val = pd.DataFrame(y_val)
y_val['uId'] = train.uId
y_val.to_csv('../../data/features/rnn_feature_train.csv',index = False)
y_test = pd.DataFrame(y_test)
y_test = test.uId
y_test.to_csv('../../data/features/rnn_feature_test.csv', index=False)

    