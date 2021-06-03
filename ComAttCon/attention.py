# coding=utf-8

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
         self.supports_masking = True
         self.init = initializers.get('uniform')

         self.W_regularizer = regularizers.get(W_regularizer)
         self.b_regularizer = regularizers.get(b_regularizer)

         self.W_constraint = constraints.get(W_constraint)
         self.b_constraint = constraints.get(b_constraint)

         self.bias = bias
         self.step_dim = step_dim
         self.features_dim = 0

         super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        # e = K.dot(x, self.W)
        # x: shape(?, ?, 128)
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        # e: shape(?, 250)
        if self.bias:
            e += self.b
        e = K.tanh(e)
        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        #a:shape=(?,250,1), x：shape=(?,?,128)
        # b = a * x
        # b:shape=(?,250,128)
        # c = K.sum(a * x, axis=1)
        b = a * x
        # c:shape=(?,128)
        return b

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[1],self.features_dim
        return input_shape[0], input_shape[1],self.features_dim


# class Attention(Layer):
#     def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
#                  u_constraint=None, b_constraint=None, use_W=True, use_bias=False, return_self_attend=False,
#                  return_attend_weight=True, **kwargs):
#         self.supports_masking = True
#
#         self.init = initializers.get('uniform')
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.u_regularizer = regularizers.get(u_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.u_constraint = constraints.get(u_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.use_W = use_W
#         self.use_bias = use_bias
#         self.return_self_attend = return_self_attend  # whether perform self attention and return it
#         self.return_attend_weight = return_attend_weight  # whether return attention weight
#         super(Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#
#         if self.use_W:
#             self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],), initializer=self.init,
#                                      name='{}_W'.format(self.name), regularizer=self.W_regularizer,
#                                      constraint=self.W_constraint)
#         if self.use_bias:
#             self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer, constraint=self.b_constraint)
#
#         self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
#                                  regularizer=self.u_regularizer, constraint=self.u_constraint)
#
#         super(Attention, self).build(input_shape)
#
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
#
#     def call(self, x, mask=None):
#         if self.use_W:
#             x = K.tanh(K.dot(x, self.W))
#
#         ait = Attention.dot_product(x, self.u)
#         if self.use_bias:
#             ait += self.b
#
#         a = K.exp(ait)
#
#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())
#
#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#
#         if self.return_self_attend:
#             attend_output = K.sum(x * K.expand_dims(a), axis=1)
#             if self.return_attend_weight:
#                 return [attend_output, a]
#             else:
#                 return attend_output
#         else:
#             return a
#
#     def compute_output_shape(self, input_shape):
#         if self.return_self_attend:
#             if self.return_attend_weight:
#                 return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
#             else:
#                 return input_shape[0], input_shape[-1]
#         else:
#             return input_shape[0], input_shape[1]
#
#     @staticmethod
#     def dot_product(x, kernel):
#         """
#         Wrapper for dot product operation, in order to be compatible with both
#         Theano and Tensorflow
#         Args:
#             x (): input
#             kernel (): weights
#         Returns:
#         """
#         if K.backend() == 'tensorflow':
#             return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
#         else:
#             return K.dot(x, kernel)
