import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os

from dataset import TFRecordCIFAR
from DenseNet import DenseNet


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_steps', '240000', 'number of steps for optimization')
tf.flags.DEFINE_integer('batch_size', '64', 'batch size for training')
tf.flags.DEFINE_integer('depth', '100', 'depth of the architecture')
tf.flags.DEFINE_integer('growth_rate', '12', 'growth rate of the architecture')
tf.flags.DEFINE_float('momentum', '0.9', 'momentum for Momentum Optimizer')
tf.flags.DEFINE_string('log_dir', 'ckpt_181005_v1', 'path to logging directory')
tf.flags.DEFINE_string('data_dir', 'data', 'path to dataset')
tf.flags.DEFINE_string('data_name', 'CIFAR-10', 'name of dataset')
tf.flags.DEFINE_string('mode', 'train', 'either train or valid')


def main(_):
    '''
     Shortcuts
    '''
    log_dir = FLAGS.log_dir
    batch_size = FLAGS.batch_size if FLAGS.mode == 'train' else 1
    num_classes = 10 if FLAGS.data_name == 'CIFAR-10' else 100

    '''
     Setting up the model
    '''    
    is_training = True if FLAGS.mode == 'train' else False
        
    dataset = TFRecordCIFAR(FLAGS.data_dir, FLAGS.data_name, num_classes)
    data, num_samples = dataset.load_batch(FLAGS.mode, batch_size)
        
    # make synonyms for data
    images = data[0]
    labels = data[1]
        
    net = DenseNet(FLAGS.depth, FLAGS.growth_rate)
    logits = net.model(images, num_classes, is_training)

    
    if FLAGS.mode == 'valid':
        saver = tf.train.Saver(slim.get_variables_to_restore())
        coord = tf.train.Coordinator()
        
        with tf.Session() as sess:
            '''
             Restore parameters from check point
            '''
            saver.restore(sess, tf.train.latest_checkpoint(log_dir))

            tf.train.start_queue_runners(sess, coord)

            acc = 0            
            time_per_image = time.time()
            for i in range(num_samples):
                r_logits, r_labels = sess.run([logits, labels])
                acc += np.argmax(r_logits) == np.argmax(r_labels)

            coord.request_stop()
            coord.join()
            
            time_per_image = (time.time() - time_per_image) / num_samples
            print('time elapsed: ' + str(time_per_image))

            acc /= num_samples
            print('Accuracy: ' + str(acc))

    elif FLAGS.mode == 'train':
        '''
         Define the loss function
        '''
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        total_loss = tf.losses.get_total_loss()

        '''
         Define summaries
        '''
        tp = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(tp, tf.float32))
        tf.summary.scalar('acc', acc)
        tf.summary.scalar('loss', loss)

        '''
         Define the learning rate
        '''
        num_batches_per_epoch = num_samples / FLAGS.batch_size
            
        lr = tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                            [int(num_batches_per_epoch * 150), int(num_batches_per_epoch * 225)],
                                            [0.1, 0.01, 0.001])

        '''
         Define the optimizer
        '''
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=FLAGS.momentum, use_nesterov=True)
    
        '''
         Training phase
        '''
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        # generate a log to save hyper-parameter info
        with open(os.path.join(log_dir, 'info.txt'), 'w') as f:
            f.write('num_steps: ' + str(FLAGS.num_steps) + '\n')
            f.write('batch_size: ' + str(FLAGS.batch_size) + '\n')
            f.write('depth: ' + str(FLAGS.depth) + '\n')
            f.write('growth_rate: ' + str(FLAGS.growth_rate) + '\n')
            f.write('momentum: ' + str(FLAGS.momentum) + '\n')
            f.write('data_dir: ' + FLAGS.data_dir + '\n')
            f.write('data_name: ' + FLAGS.data_name + '\n')
            f.write('mode: ' + FLAGS.mode + '\n')

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(
                train_op = train_op,
                logdir = log_dir,
                init_fn = None,
                number_of_steps = FLAGS.num_steps,
                summary_op = tf.summary.merge_all())

        print('Finished training. Final batch loss %f' %final_loss)

    else:
        print('Unknown mode')


if __name__ == "__main__":
    tf.app.run()