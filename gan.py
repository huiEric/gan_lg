import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

init_time = str(time.asctime()).replace(' ', '-')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('iters', '100000', 'Number of loop iterations')
tf.app.flags.DEFINE_integer('batch', '32', 'Batch size')
tf.app.flags.DEFINE_integer('z_dim', '81', 'Dimision of z')
tf.app.flags.DEFINE_integer('k_steps', '7', 'Number of iterations to train the discriminator for every loop iteration')
tf.app.flags.DEFINE_float('lr', '1e-3', 'Learning rate')
tf.app.flags.DEFINE_string('filename', 'log_'+init_time+'.txt', 'the log file for the experiment records')

filename = os.path.join('./', FLAGS.filename)
z_dim = FLAGS.z_dim


if not os.path.exists('out/'):
    os.makedirs('out/')

def Log(string, log_file_path=filename):
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def D(X):
    theta_D = tf.get_collection('D_var')[0]
    (D_W1, D_W2, D_b1, D_b2) = (theta_D[0], theta_D[1], theta_D[2], theta_D[3])
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def G(Z):
    theta_G = tf.get_collection('G_var')[0]
    (G_W1, G_W2, G_b1, G_b2) = (theta_G[0], theta_G[1], theta_G[2], theta_G[3])
    G_h1 = tf.nn.relu(tf.matmul(Z, G_W1) + G_b1)
    G_logit = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_logit)

    return G_prob


def main(args):
    ############################################################################################################################
    #       1. Loading Data                                                                                                    #
    ############################################################################################################################
    Log('Loading MNIST dataset...')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)    

   
    ############################################################################################################################
    #       2. Hyper-param Setting                                                                                             #
    ############################################################################################################################
    Log('Setting Hyper-param...')
    lr = FLAGS.lr
    iters = FLAGS.iters
    k_steps = FLAGS.k_steps
    batch_size = FLAGS.batch
    Log('Learning rate: {}'.format(lr))
    Log('Num of iterations: {}'.format(iters))
    Log('Batch size: {}'.format(batch_size))
    Log('Dimension of z: {}'.format(z_dim))    
    # Log(str(sys.argv))


    ############################################################################################################################
    #       3. Model Setup                                                                                                     #
    ############################################################################################################################

    # 3.1 Input #
    X = tf.placeholder(shape=[None, 784], dtype=tf.float32 )
    Z = tf.placeholder(shape=[None, z_dim], dtype=tf.float32 )
    
    D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

    theta_D = [D_W1, D_W2, D_b1, D_b2]
    tf.add_to_collection('D_var', theta_D)

    G_W1 = tf.Variable(xavier_init([z_dim, 128]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')    

    G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

    theta_G = [G_W1, G_W2, G_b1, G_b2]
    tf.add_to_collection('G_var', theta_G)
 
    
    # 3.2 Loss function #
    G_sample = G(Z)
    D_real, D_logit_real = D(X)
    D_fake, D_logit_fake = D(G_sample)

    # D_loss = 0.5 * (tf.reduce_mean((1 - D_real)**2) + tf.reduce_mean(D_fake**2))
    # G_loss = 0.5 * tf.reduce_mean((1 - D_fake)**2)
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))
    
    # 3.3 Optimizer #
    opt_adm = tf.train.AdamOptimizer()  # """Here you need to use Adam optimizer with no learning rate specified"""
    opt_sgd = tf.train.GradientDescentOptimizer(learning_rate=lr)  # """Here you need to use gradient descent optimizer with a learning rate specified"""
    # opt_adm = tf.train.AdamOptimizer(learning_rate=lr)
    # opt_sgd = tf.train.AdamOptimizer(learning_rate=lr)

    # 3.4 Training step #
    train_G_step = opt_adm.minimize(G_loss, var_list=tf.get_collection('G_var')[0])
    train_D_step = opt_sgd.minimize(D_loss, var_list=tf.get_collection('D_var')[0])
    

    ############################################################################################################################
    #       4. Training                                                                                                        #
    ############################################################################################################################
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 4.1 Initialization #
        sess.run(tf.global_variables_initializer())
        i = 0
        
        # 4.2 Model traininig #
        for it in range(iters):
            # 4.2.1 Do sampling 
            # X_batch, _ = mnist.train.next_batch(batch_size)

            # 4.2.2 Train the discriminator #
            for k in range(k_steps):
                X_batch, _ = mnist.train.next_batch(batch_size)
                z_batch = sample_z(batch_size, z_dim)
                _, D_loss_curr = sess.run([train_D_step, D_loss], feed_dict={X: X_batch, Z: z_batch}) 


            # 4.2.3 Train the generator #
            X_batch, _ = mnist.train.next_batch(batch_size)
            z_batch = sample_z(batch_size, z_dim)
            _, G_loss_curr = sess.run([train_G_step, G_loss], feed_dict={X: X_batch, Z: z_batch})        

            if it % 1000 == 0:
                Log('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

                samples = sess.run(G_sample, feed_dict={Z: sample_z(16, z_dim)})

                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
             

    ############################################################################################################################
    #       5. Testing                                                                                                         #
    ############################################################################################################################
    """You should generate some fake pictures"""
    # if not os.path.exists('out/'):
    #     os.makedirs('out/')
    # print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(iters, D_loss_curr, G_loss_curr))
    # with tf.Session() as sess:
    #     samples = sess.run(G_sample, feed_dict={Z: sample_z(16, z_dim)})
    # fig = plot(samples)
    # plt.savefig('out/out.png')
    # plt.close(fig)
    

if __name__ == '__main__':
    tf.app.run()
