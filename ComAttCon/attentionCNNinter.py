# coding=utf-8

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer

# class Attention_zw(Layer):
#     def __init__(self, step_dim,
#                  W_regularizer=None, b_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         self.supports_masking = True
#         self.initializer = initializers.get('uniform')
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.bias = bias
#         self.step_dim = step_dim
#         self.features_dim = 0
#
#         super(Attention_zw, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         context_shape, asp_text_shape = input_shape
#         self.context_w = self.add_weight(shape=(context_shape[-1], asp_text_shape[-1]), initializer=self.initializer,
#                                          regularizer=self.W_regularizer, constraint=self.W_constraint,
#                                          name='{}_context_w'.format(self.name))
#         self.context_b = self.add_weight(shape=(context_shape[1],), initializer='zero', regularizer=self.b_regularizer,
#                                          constraint=self.b_constraint, name='{}_context_b'.format(self.name))
#         self.aspect_w = self.add_weight(shape=(asp_text_shape[-1], context_shape[-1]), initializer=self.initializer,
#                                         regularizer=self.W_regularizer, constraint=self.W_constraint,
#                                         name='{}_aspect_w'.format(self.name))
#         self.aspect_b = self.add_weight(shape=(asp_text_shape[1],), initializer='zero', regularizer=self.b_regularizer,
#                                         constraint=self.b_constraint, name='{}_aspect_b'.format(self.name))
#
#
#         # self.W = self.add_weight((input_shape[-1],),
#         #                          initializer=self.init,
#         #                          name='{}_W'.format(self.name),
#         #                          regularizer=self.W_regularizer,
#         #                          constraint=self.W_constraint)
#         # self.features_dim = input_shape[-1]
#         #
#         # if self.bias:
#         #     self.b = self.add_weight((input_shape[1],),
#         #                              initializer='zero',
#         #                              name='{}_b'.format(self.name),
#         #                              regularizer=self.b_regularizer,
#         #                              constraint=self.b_constraint)
#         # else:
#         #     self.b = None
#
#         self.built = True
#
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
#
#     def call(self, inputs, mask=None):
#         assert isinstance(inputs, list)
#         if mask is not None:
#             context_mask, asp_text_mask = mask
#         else:
#             context_mask = None
#             asp_text_mask = None
#
#         context, asp_text = inputs
#
#         context_avg = K.mean(context, axis=1)
#         asp_text_avg = K.mean(asp_text, axis=1)
#
#         # attention over context with aspect_text
#         cc = K.dot(context, self.context_w)
#         a_c = K.tanh(K.batch_dot(asp_text_avg, cc, axes=[1, 2]) + self.context_b)
#         a_c = K.exp(a_c)
#         if context_mask is not None:
#             a_c *= K.cast(context_mask, K.floatx())
#         a_c /= K.cast(K.sum(a_c, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#         attend_context = K.sum(context * K.expand_dims(a_c), axis=1)
#         # attend_concat = K.concatenate([attend_context, attend_context], axis=-1)
#
#         return attend_context
#
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         context_shape, asp_text_shape = input_shape
#         return context_shape[0],context_shape[-1]

class InteractiveAttentionCNN(Layer):
    """
    Interactive attention between context and aspect text.
    Supporting Masking.
    Follows the work of Dehong et al. [https://www.ijcai.org/proceedings/2017/0568.pdf]
    "Interactive Attention Networks for Aspect-Level Sentiment Classification"
    """

    def __init__(self, return_attend_weight=False, initializer='uniform', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(InteractiveAttentionCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape

        self.context_w = self.add_weight(shape=(context_shape[-1], asp_text_shape[-1]), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint,
                                         name='{}_context_w'.format(self.name))
        self.context_b = self.add_weight(shape=(context_shape[1],), initializer='zero', regularizer=self.regularizer,
                                         constraint=self.constraint, name='{}_context_b'.format(self.name))

        self.built = True
        # super(InteractiveAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        if mask is not None:
            context_mask, asp_text_mask
        else:
            context_mask = None
            asp_text_mask = None

        #context:tensor,shape(?,?,128)
        context, asp_text = inputs
        # c= K.reshape(context, (-1, context[-1]))
        # context_avg = K.mean(context, axis=1)
        # # context_avg:shape= (?,128)
        # asp_text_avg = K.mean(asp_text, axis=1)
        a_c = K.tanh(K.batch_dot(asp_text, K.dot(context, self.context_w), axes=[1, 2]) + self.context_b)
        a_c = K.exp(a_c)
        if context_mask is not None:
            a_c *= K.cast(context_mask, K.floatx())
        a_c /= K.cast(K.sum(a_c, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.max(context * K.expand_dims(a_c), axis=1)
        # attend_context=context * K.expand_dims(a_c)

        # attend_concat = K.concatenate([attend_context, attend_asp_text], axis=-1)
        attend_concat = attend_context
        # attend_concat = K.concatenate([attend_context, context_avg], axis=-1)

        # attend_concat = K.concatenate([context_avg, context_avg], axis=-1)

        if self.return_attend_weight == True:
            return [attend_concat, a_c]
        else:
            return attend_concat


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape

        if self.return_attend_weight == True:
            return [(context_shape[0], context_shape[-1] ), (context_shape[0], context_shape[1]),
                    (asp_text_shape[0], asp_text_shape[1])]
        # else:
        #     return context_shape[0], context_shape[1],context_shape[-1]
        else:
            return context_shape[0], context_shape[-1]