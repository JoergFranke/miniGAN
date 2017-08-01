


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from miniGAN.mnist_batch_generator import MNIST_loader
from miniGAN.conv_layer import conv_layer, deconv_layer, maxp_layer, sigmoid_layer, conv_norm_acti_drop_layer



session = 2
session_path = "expt_miniGan_{}".format(session)
data_path = 'tmp_data/'

if not os.path.isdir(session_path):
    os.mkdir(session_path)
if not os.path.isdir(data_path):
    os.mkdir(data_path)

batch_size = 10
EPOCHS = 500

loader = MNIST_loader(data_path, batch_size)
train_batch_gen, valid_batch_gen = loader.get_batch_loader(max_perc=0.8)

keep_prob = 0.96

'''
############################
###       GENERATOR      ###
############################
'''


def generator(image):
    # conv 0
    g_conv01 = conv_norm_acti_drop_layer(image, [3,3],  16, "g_conv01", stride=1, padding='SAME', norm=True, keep_prob=keep_prob)
    g_conv02 = conv_norm_acti_drop_layer(g_conv01  , [3,3],  16, "g_conv02", stride=1, padding='SAME', keep_prob=keep_prob)
    g_maxp10 = maxp_layer(g_conv02, 'maxp10', pooling=2) # 28x28 -> 14x14

    # conv 1
    g_conv11 = conv_norm_acti_drop_layer(g_maxp10,   [3,3],  16, "g_conv11", stride=1, padding='SAME', keep_prob=keep_prob)
    g_conv12 = conv_norm_acti_drop_layer(g_conv11  , [3,3],  16, "g_conv12", stride=1, padding='SAME', keep_prob=keep_prob)
    g_maxp20 = maxp_layer(g_conv12, 'g_maxp20', pooling=2) # 14x14 -> 7x7

    #conv 2
    g_conv21 = conv_norm_acti_drop_layer(g_maxp20,   [3,3],  16, "g_conv21", stride=1, padding='SAME', keep_prob=keep_prob)
    g_conv22 = conv_norm_acti_drop_layer(g_conv21  , [3,3],  16, "g_conv22", stride=1, padding='SAME', keep_prob=keep_prob)

    #deconv 2
    g_decov2 = deconv_layer(g_conv22, [2,2], 16, 'g_decov2', stride=2, padding='SAME')
    g_concat2 = tf.concat([g_conv12,g_decov2, ], axis=3)
    g_conv31 = conv_norm_acti_drop_layer(g_concat2,   [3,3],  16, "g_conv31", stride=1, padding='SAME', keep_prob=keep_prob)

    #deconv 1
    g_decov1 = deconv_layer(g_conv31, [2,2], 16, 'gdecov1', stride=2, padding='SAME')
    g_concat1 = tf.concat([g_conv02,g_decov1, ], axis=3)
    g_conv41 = conv_norm_acti_drop_layer(g_concat1,   [3,3],  16, "gconv41", stride=1, padding='SAME', keep_prob=keep_prob)

    g_prediction, g_flat_pred = sigmoid_layer(g_conv41, "g_sigmoid", weighted=True, num_classes=loader.image_channels, pre_depth=16)

    return g_prediction



'''
############################
###    DISCRIMINATOR     ###
############################
'''


def discriminator(seg, reuse):

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        # conv 0
        d_conv01 = conv_layer(seg,  [3,3],  16, "d_conv01", stride=1, padding='SAME')
        d_conv02 = conv_layer(d_conv01  , [3,3],  16, "d_conv02", stride=1, padding='SAME')

        d_maxp10 = maxp_layer(d_conv02, 'd_maxp10', pooling=2) # 28x28 -> 14x14
        # conv 1
        d_conv11 = conv_layer(d_maxp10,   [3,3],  16, "d_conv11", stride=1, padding='SAME')
        d_conv12 = conv_layer(d_conv11  , [3,3],  16, "d_conv12", stride=1, padding='SAME')

        d_maxp20 = maxp_layer(d_conv12, 'd_maxp20', pooling=2) # 14x14 -> 7x7
        #conv 2
        d_conv21 = conv_layer(d_maxp20,   [3,3],  16, "d_conv21", stride=1, padding='SAME')
        d_conv22 = conv_layer(d_conv21  , [3,3],  16, "d_conv22", stride=1, padding='SAME')
        #global average pool
        d_gap = tf.nn.avg_pool(d_conv22, ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME')

        d_prediction, _ = sigmoid_layer(d_gap, "d_sigmoid", weighted=True, num_classes=1, pre_depth=16)
        d_prediction = tf.squeeze(d_prediction)

        return d_prediction




'''
#### LOSSES
'''


x_img = tf.placeholder(tf.float32, shape=[batch_size, loader.image_size,loader.image_size,loader.image_channels])
y_seg = tf.placeholder(tf.float32, shape=[batch_size, loader.image_size,loader.image_size,loader.image_channels])

real_seg = discriminator(y_seg, reuse=False)

seg_pred = generator(x_img)
fake_seg = discriminator(seg_pred, reuse=True)

gen_loss = -(tf.log(fake_seg))
disc_loss = -(tf.log(real_seg) + tf.log(1-fake_seg))

'''
##### COUNT PARAMETER
'''
t_vars = tf.trainable_variables()
gen_var_list = [var for var in t_vars if 'g_' in var.name ]
disc_var_list = [var for var in t_vars if 'd_' in var.name ]


gen_parameters = 0
for variable in gen_var_list:
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    gen_parameters += variable_parametes
print("generator parameters: ")
print(gen_parameters)

dist_parameters = 0
for variable in disc_var_list:
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    dist_parameters += variable_parametes
print("discriminator parameters: ")
print(dist_parameters)


'''
##### OPTIMIZER
'''
train_gen = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(gen_loss, var_list=gen_var_list)
train_disc = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(disc_loss, var_list=disc_var_list)


'''
##### TRAINING
'''
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init_op)

    for e in range(EPOCHS):

        d_loss, g_loss, fs_log, rs_log, count = 0, 0, 0, 0, 0
        for step in tqdm(range( loader.no_train_batches )):

            """ train discriminator """
            batch = next(train_batch_gen)
            _, dl, fs, rs = sess.run([train_disc, disc_loss, fake_seg, real_seg], feed_dict={x_img: batch['data'], y_seg: batch['seg']})

            """ train generator """
            batch = next(train_batch_gen)
            _, gl = sess.run([train_gen, gen_loss], feed_dict={x_img: batch['data'], y_seg: batch['seg']})

            d_loss += np.mean(dl)
            g_loss += np.mean(gl)
            fs_log += np.mean(fs)
            rs_log += np.mean(rs)
            count += 1
        print("epoch {:2.0f}, d loss {:0.4f}, g loss {:0.4f}, real seg {:0.4f}, fake seg {:0.4f}".format(e, d_loss/count, g_loss/count, fs_log/count, rs_log/count))


        if e % 30 == 0:
            """ print generator output """
            pred_img = sess.run([seg_pred], feed_dict={x_img: batch['data']})[0]
            fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(12, 8))
            for i in range(3):
                axes[i, 0].imshow(batch['data'][i,:,:,0])
                axes[i, 1].imshow(batch['seg'][i,:,:,0])
                axes[i, 2].imshow(pred_img[i,:,:,0])
                axes[i, 0].get_xaxis().set_visible(False)
                axes[i, 1].get_xaxis().set_visible(False)
                axes[i, 2].get_xaxis().set_visible(False)
                axes[i, 0].get_yaxis().set_visible(False)
                axes[i, 1].get_yaxis().set_visible(False)
                axes[i, 2].get_yaxis().set_visible(False)
            cols = ['Noisy', 'Truth', 'Generator']
            for ax, col in zip(axes[0], cols):
                ax.set_title(col)
            plt.savefig(os.path.join(session_path, 'save_img_{}.png'.format(e)))

            save_path = saver.save(sess, os.path.join(session_path, "tensorflow_dump_{}.ckpt".format(e)))

            """ get MSE of validation set """
            mse = 0
            for step in range(loader.no_valid_batches):
                batch = next(valid_batch_gen)
                pred = sess.run(seg_pred, feed_dict={x_img: batch['data']})
                pred = pred[:,:,:,0]
                truth = batch['seg'][:,:,:,0]
                mse += mean_squared_error(truth.flatten(), pred.flatten())
            print("epoch {:2.0f}, MSE {:0.4f}".format(e, mse/loader.no_valid_batches))

