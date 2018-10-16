import tensorflow as tf
import tensorflow.contrib.slim as slim

class DenseNet:
    
    def __init__(self, dataset, depth, growth_rate, num_blocks=None, init_conv_out=None, bc_mode=True, reduction_rate=0.5):
        
        self.dataset = dataset
        self.depth = depth
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.init_conv_out = init_conv_out
        self.bc_mode = bc_mode
        self.reduction_rate = reduction_rate

        self._adapt_params_to_dataset()


    def _adapt_params_to_dataset(self):
                
        if self.dataset == 'CIFAR-10' or self.dataset == 'CIFAR-100':
            self.num_blocks = int((self.depth - 4) / 3)
            if self.init_conv_out == None:
                self.init_conv_out = 2 * self.growth_rate
            if self.bc_mode:
                self.num_blocks = int(self.num_blocks / 2)
            else:
                self.reduction_rate = 1.0
                
        if self.dataset == 'ImageNet':
            self.growth_rate = 32
            self.init_conv_out = 2 * self.growth_rate
            if self.depth == 121:
                self.num_blocks = [6, 12, 24, 16]
            if self.depth == 169:
                self.num_blocks = [6, 12, 32, 32]
            if self.depth == 201:
                self.num_blocks = [6, 12, 48, 32]
            if self.depth == 264:
                self.num_blocks = [6, 12, 64, 48]
    

    def model(self, images, num_classes, is_training, keep_prob=1.0):
        
        self.keep_prob = keep_prob

        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_regularizer=slim.l2_regularizer(1e-4)):
            with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu, is_training=is_training):
                
                if self.dataset == 'CIFAR-10' or self.dataset == 'CIFAR-100':
                    net = slim.conv2d(images, self.init_conv_out, 3)

                    net = self._dense_block(net, self.num_blocks)
                    net = self._transition_block(net, int(net.shape[3].value * self.reduction_rate))

                    net = self._dense_block(net, self.num_blocks)
                    net = self._transition_block(net, int(net.shape[3].value * self.reduction_rate))

                    net = self._dense_block(net, self.num_blocks)
                    net = self._transition_block(net, net.shape[3], True, 8)

                if self.dataset == 'ImageNet':
                    net = slim.conv2d(images, self.init_conv_out, 7, 2)
                    net = slim.batch_norm(net)
                    net = slim.max_pool2d(net, 3, padding='SAME')

                    net = self._dense_block(net, self.num_blocks[0])
                    net = self._trainsition_block(net, int(net.shape[3].value * self.reduction_rate))

                    net = self._dense_block(net, self.num_blocks[1])
                    net = self._trainsition_block(net, int(net.shape[3].value * self.reduction_rate))

                    net = self._dense_block(net, self.num_blocks[2])
                    net = self._trainsition_block(net, int(net.shape[3].value * self.reduction_rate))

                    net = self._dense_block(net, self.num_blocks[3])
                    net = self._transition_block(net, net.shape[3], True, 7, 1)
                
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

    
    def _transition_block(self, net, num_outputs, last=False, pool_size=8, stride=2):

        net = slim.batch_norm(net)
        if last:
            net = slim.avg_pool2d(net, pool_size, stride)
        else:
            net = self._conv2d_layer(net, num_outputs, 1)
            net = slim.avg_pool2d(net, 2)

        return net