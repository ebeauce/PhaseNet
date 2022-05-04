import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np
import logging

class ModelConfig():

    batch_size = 20
    depths = 5
    filters_root = 8
    kernel_size = [7, 1]
    pool_size = [4, 1]
    dilation_rate = [1, 1]
    class_weights = [1.0, 1.0, 1.0]
    loss_type = "cross_entropy"
    summary = True

    X_shape = [3000, 1, 3]
    n_channel = X_shape[-1]
    Y_shape = [3000, 1, 3]
    n_class = Y_shape[-1]

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def update_args(self, args):
        for k,v in vars(args).items():
            setattr(self, k, v)


def crop_and_concat(net1, net2):
    """
    the size(net1) <= size(net2)
    """

    ## dynamic shape
    chn1 = net1.get_shape().as_list()[-1]
    chn2 = net2.get_shape().as_list()[-1]
    net1_shape = tf.shape(net1)
    net2_shape = tf.shape(net2)
    offsets = [0, (net2_shape[1] - net1_shape[1]) // 2,
            (net2_shape[2] - net1_shape[2]) // 2, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)

    out = tf.concat([net1, net2_resize], 3)
    out.set_shape([None, None, None, chn1+chn2])

    return out 


def crop_only(net1, net2):
    """
    the size(net1) <= size(net2)
    """
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
    offsets = [0, (net2_shape[1] - net1_shape[1]) // 2,
            (net2_shape[2] - net1_shape[2]) // 2, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    return net2_resize

class UNet:
    """UNet model from PhaseNet in prediction mode.  

    """
    def __init__(self, config=ModelConfig(), input_batch=None):
        self.depths = config.depths
        self.filters_root = config.filters_root
        self.kernel_size = config.kernel_size
        self.dilation_rate = config.dilation_rate
        self.pool_size = config.pool_size
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.batch_size = config.batch_size

        self.build(input_batch)

    def add_placeholders(self, input_batch=None):
        if input_batch is None:
            self.X = tf.compat.v1.placeholder(
                    dtype=tf.float32, shape=[None, None, None, self.X_shape[-1]],
                    name='X')
            self.Y = tf.compat.v1.placeholder(
                    dtype=tf.float32, shape=[None, None, None, self.n_class],
                    name='y')
        else:
          self.X = input_batch[0]
          self.input_batch = input_batch

        self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name="is_training")
        self.drop_rate = tf.compat.v1.placeholder(dtype=tf.float32, name="drop_rate")


    def add_prediction_op(self):
        #logging.info("Model: depths {depths}, filters {filters}, "
        #       "filter size {kernel_size[0]}x{kernel_size[1]}, "
        #       "pool size: {pool_size[0]}x{pool_size[1]}, "
        #       "dilation rate: {dilation_rate[0]}x{dilation_rate[1]}".format(
        #        depths=self.depths,
        #        filters=self.filters_root,
        #        kernel_size=self.kernel_size,
        #        dilation_rate=self.dilation_rate,
        #        pool_size=self.pool_size))

        self.regularizer = None

        self.initializer = tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform")

        # down sample layers
        convs = [None] * self.depths # store output of each depth

        with tf.compat.v1.variable_scope("Input"):
            net = self.X
            net = tf.compat.v1.layers.Conv2D(
                         self.filters_root,
                         self.kernel_size,
                         activation=None,
                         padding='same',
                         dilation_rate=self.dilation_rate,
                         kernel_initializer=self.initializer,
                         kernel_regularizer=self.regularizer,
                         name="input_conv")(net)
            net = tf.compat.v1.layers.BatchNormalization(
                              trainable=False,
                              name="input_bn")(net)
            net = tf.nn.relu(net,
                     name="input_relu")
            # net = tf.nn.dropout(net, self.keep_prob)
            #net = tf.compat.v1.layers.dropout(net,
            #            rate=self.drop_rate,
            #            training=self.is_training,
            #            name="input_dropout")


        for depth in range(0, self.depths):
            with tf.compat.v1.variable_scope("DownConv_%d" % depth):
                filters = int(2**(depth) * self.filters_root)

                net = tf.compat.v1.layers.Conv2D(
                             filters,
                             self.kernel_size,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             dilation_rate=self.dilation_rate,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.regularizer,
                             name="down_conv1_{}".format(depth + 1))(net)
                net = tf.compat.v1.layers.BatchNormalization(
                                  trainable=False,
                                  name="down_bn1_{}".format(depth + 1))(net)
                net = tf.nn.relu(net,
                         name="down_relu1_{}".format(depth+1))
                #net = tf.compat.v1.layers.dropout(net,
                #            training=self.is_training,
                #            name="down_dropout1_{}".format(depth + 1))

                convs[depth] = net

                if depth < self.depths - 1:
                  net = tf.compat.v1.layers.Conv2D(
                               filters,
                               self.kernel_size,
                               strides=self.pool_size,
                               activation=None,
                               use_bias=False,
                               padding='same',
                               dilation_rate=self.dilation_rate,
                               kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer,
                               name="down_conv3_{}".format(depth + 1))(net)
                  net = tf.compat.v1.layers.BatchNormalization(
                                    trainable=False,
                                    name="down_bn3_{}".format(depth + 1))(net)
                  net = tf.nn.relu(net,
                           name="down_relu3_{}".format(depth+1))
                  #net = tf.compat.v1.layers.dropout(net,
                  #          rate=self.drop_rate,
                  #          training=self.is_training,
                  #          name="down_dropout3_{}".format(depth + 1))


        # up layers
        for depth in range(self.depths - 2, -1, -1):
            with tf.compat.v1.variable_scope("UpConv_%d" % depth):
                filters = int(2**(depth) * self.filters_root)
                net = tf.compat.v1.layers.Conv2DTranspose(
                                 filters,
                                 self.kernel_size,
                                 strides=self.pool_size,
                                 activation=None,
                                 use_bias=False,
                                 padding="same",
                                 kernel_initializer=self.initializer,
                                 kernel_regularizer=self.regularizer,
                                 name="up_conv0_{}".format(depth+1))(net)
                net = tf.compat.v1.layers.BatchNormalization(
                                  trainable=False,
                                  name="up_bn0_{}".format(depth + 1))(net)
                net = tf.nn.relu(net,
                         name="up_relu0_{}".format(depth+1))
                #net = tf.compat.v1.layers.dropout(net,
                #            rate=self.drop_rate,
                #            training=self.is_training,
                #            name="up_dropout0_{}".format(depth + 1))

                
                #skip connection
                net = crop_and_concat(convs[depth], net)

                net = tf.compat.v1.layers.Conv2D(
                             filters,
                             self.kernel_size,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             dilation_rate=self.dilation_rate,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.regularizer,
                             name="up_conv1_{}".format(depth + 1))(net)
                net = tf.compat.v1.layers.BatchNormalization(
                                  trainable=False,
                                  name="up_bn1_{}".format(depth + 1))(net)
                net = tf.nn.relu(net,
                         name="up_relu1_{}".format(depth + 1))
                #net = tf.compat.v1.layers.dropout(net,
                #            rate=self.drop_rate,
                #            training=self.is_training,
                #            name="up_dropout1_{}".format(depth + 1))


        # Output Map
        with tf.compat.v1.variable_scope("Output"):
            net = tf.compat.v1.layers.Conv2D(
                         self.n_class,
                         (1, 1),
                         activation=None,
                         padding='same',
                         kernel_initializer=self.initializer,
                         kernel_regularizer=self.regularizer,
                         name="output_conv")(net)
            output = net
         
        with tf.compat.v1.variable_scope("representation"):
            self.representation = convs[-1]

        #with tf.compat.v1.variable_scope("logits"):
        #    self.logits = output
        #    tmp = tf.compat.v1.summary.histogram("logits", self.logits)
        #    self.summary_train.append(tmp)

        with tf.compat.v1.variable_scope("preds"):
            self.preds = tf.nn.softmax(output)
            #tmp = tf.compat.v1.summary.histogram("preds", self.preds)
            #self.summary_train.append(tmp)

    def build(self, input_batch=None):
        self.add_placeholders(input_batch)
        self.add_prediction_op()
        return 0
