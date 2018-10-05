import tensorflow as tf
import tensorflow.contrib.slim as slim

class DenseNet:
    
    def __init__(self, depth, growth_rate, num_channels=None, bc_mode=True, reduction_rate=0.5, keep_prob=1.0):
        
        self.depth = depth
        self.num_blocks = int((depth - 4) / 3)
        self.growth_rate = growth_rate
        if num_channels == None:
            num_channels = 2 * growth_rate
        self.num_channels = num_channels
        self.bc_mode = bc_mode
        if bc_mode:
            self.num_blocks = int(self.num_blocks / 2)
            self.reduction_rate = reduction_rate
        else:
            self.reduction_rate = 1.0
        self.keep_prob = keep_prob
    

    def model(self, images, num_classes, is_training):
        
        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_regularizer=slim.l2_regularizer(1e-4)):
            with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d(images, self.num_channels, 3)

                net = self._dense_block(net, self.num_blocks)
                net = self._transition_block(net, int(net.shape[3].value * self.reduction_rate))

                net = self._dense_block(net, self.num_blocks)
                net = self._transition_block(net, int(net.shape[3].value * self.reduction_rate))

                net = self._dense_block(net, self.num_blocks)
                net = self._transition_block(net, net.shape[3], True, 8)
                net = slim.conv2d(net, num_classes, 1)
                net = tf.squeeze(net)

        return net

        
    def _conv2d_layer(self, net, num_outputs, kernel_size):

        net = slim.conv2d(net, num_outputs, kernel_size)
        if self.keep_prob < 1.0:
            net = slim.dropout(net, self.keep_prob)

        return net

    
    def _dense_connect_layer(self, net):
        
        net = slim.batch_norm(net)
        if self.bc_mode:
            net = self._conv2d_layer(net, 4 * self.growth_rate, 1)
            net = slim.batch_norm(net)
        net = self._conv2d_layer(net, self.growth_rate, 3)

        return net

    
    def _dense_block(self, net, num_blocks):
        
        for i in range(0, num_blocks):
            out_net = self._dense_connect_layer(net)
            net = tf.concat([net, out_net], 3)

        return net

    
    def _transition_block(self, net, num_outputs, last = False, pool_size=8):

        net = slim.batch_norm(net)
        if last:
            net = slim.avg_pool2d(net, pool_size)
        else:
            net = self._conv2d_layer(net, num_outputs, 1)
            net = slim.avg_pool2d(net, 2)

        return net