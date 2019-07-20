import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
import gc
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

from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,BatchNormalization

from keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

    inputs = Input(shape=(170,))
    emb = Embedding(21099, 300,  trainable=True)(inputs)

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
    
#     fc = Dense(128, activation='sigmoid')(x)
    outputs = Dense(6, activation='softmax')(x)#   , kernel_regularizer=l2(l2_penalty), activity_regularizer=l2(l2_penalty)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Nadam',metrics =['accuracy'])
    return model   

def mlp_v2():
    model = Sequential()
    model.add(Dense(2048, input_shape=(21099,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
#     model.add(Dense(1024))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))   
#     model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    model.add(BatchNormalization())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    return model
def RnnVersion2(n_recurrent=50, n_dense=50, word_embedding_matrix= None, n_filters=50,dropout_rate=0.2, l2_penalty=0.0001,
                n_capsule = 10, n_routings = 5, capsule_dim = 16,max_len = 170, emb_size = 21099):
    K.clear_session()
    
    def conv_block(x, n, kernel_size):
        x = Conv1D(n, kernel_size, activation='relu') (x)
        x = Conv1D(n_filters, kernel_size, activation='relu') (x)
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])   
    def att_max_avg_pooling(x):
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])

    inputs = Input(shape=(max_len,))
    emb = Embedding(emb_size, 300,trainable=True)(inputs)

    # model 0
    x0 = SpatialDropout1D(dropout_rate)(emb)
    s0 = Bidirectional(
        CuDNNGRU(2*n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x0)
    x0 = att_max_avg_pooling(s0)

    # model 1
    x1 = SpatialDropout1D(dropout_rate)(emb)
    s1 = Bidirectional(
        CuDNNGRU(2*n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x1)
    x1 = att_max_avg_pooling(s1)
    
    # combine sequence output
    x = concatenate([s0, s1])
#     x = att_max_avg_pooling(x)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True, 
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    x = att_max_avg_pooling(x)
    


    # combine it all
    x = concatenate([x,x0, x1],name = 'concatenate')

    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics =['accuracy'])
    return model



def RnnVersion3(n_recurrent=50, n_dense=50, word_embedding_matrix= None, n_filters=50,dropout_rate=0.2, l2_penalty=0.0001,
               n_capsule = 10, n_routings = 5, capsule_dim = 16,):
    K.clear_session()
    
    def conv_block(x, n, kernel_size):
        x = Conv1D(n, kernel_size, activation='relu') (x)
        x = Conv1D(n_filters, kernel_size, activation='relu') (x)
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])   
    def att_max_avg_pooling(x):
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])

    input1_= Input(shape=(170, ), name='input1')
    input2_ = Input(shape=(433, ), name='input2')
    emb = Embedding(21099, 300,trainable=True)(input1_)

    # model 0
    x0 = SpatialDropout1D(dropout_rate)(emb)
    s0 = Bidirectional(
        CuDNNGRU(2*n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x0)
    x0 = att_max_avg_pooling(s0)

    # model 1
    x1 = SpatialDropout1D(dropout_rate)(emb)
    s1 = Bidirectional(
        CuDNNGRU(2*n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x1)
    x1 = att_max_avg_pooling(s1)
    
    # combine sequence output
    x = concatenate([s0, s1])
#     x = att_max_avg_pooling(x)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True, 
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    x = att_max_avg_pooling(x)
    

    # combine it all
    x = concatenate([x,x0, x1,input2_],name = 'concatenate')
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)       
 #   fc = Dense(120, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs=[input1_,input2_], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics =['accuracy'])
    return model

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


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




class GetBest(Callback):
    """Get the best model at the end of training.
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)

class AMSgrad(Optimizer):
  """AMSGrad optimizer.
  Default parameters follow those provided in the Adam paper.
  # Arguments
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.
  # References
      - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
      - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
  """
 
  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., **kwargs):
    super(AMSgrad, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(decay, name='decay')
    self.epsilon = epsilon
    self.initial_decay = decay

  
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

    t = K.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /(1. - K.pow(self.beta_1, t)))

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params] 
    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      vhat_t = K.maximum(vhat, v_t)
      p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))
      self.updates.append(K.update(vhat, vhat_t))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(K.update(p, new_p))
    return self.updates

  def get_config(self):
      config = {'lr': float(K.get_value(self.lr)),
                'beta_1': float(K.get_value(self.beta_1)),
                'beta_2': float(K.get_value(self.beta_2)),
                'decay': float(K.get_value(self.decay)),
                'epsilon': self.epsilon}
      base_config = super(AMSgrad, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))


def CapsuleNet(n_capsule = 10, n_routings = 5, capsule_dim = 16,
     n_recurrent=100, dropout_rate=0.2, l2_penalty=0.0001):
    K.clear_session()

    inputs = Input(shape=(170,))
    x = Embedding(21099, 300,  trainable=True)(inputs)        
    x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    x = PReLU()(x)
    x = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(x)
    x = Flatten(name = 'concatenate')(x)
    x = Dropout(dropout_rate)(x)
#     fc = Dense(128, activation='sigmoid')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model
	
def CapsuleNet_v2(n_capsule = 10, n_routings = 5, capsule_dim = 16,
     n_recurrent=100, dropout_rate=0.2, l2_penalty=0.0001):
    K.clear_session()

    inputs = Input(shape=(200,))
    x = Embedding(20000, 300,  trainable=True)(inputs)        
    x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    x = PReLU()(x)
    x = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(x)
    x = Flatten(name = 'concatenate')(x)
    x = Dropout(dropout_rate)(x)
#     fc = Dense(128, activation='sigmoid')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model