import tensorflow as tf
CIFAR_DIR = 'cifar-10-batches-py/'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict
dirs = ['batches.meta','data_batch_1',
        'data_batch_2','data_batch_3',
        'data_batch_4','data_batch_5',
        'test_batch']
all_data = [0,1,2,3,4,5,6]
for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)
    batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

import numpy as np

X = data_batch1[b"data"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

def one_hot_encode(vec, vals=10):
    
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():
    def __init__(self):
        self.i = 0
        self.all_train_batches = [
            data_batch1,data_batch2,
            data_batch3,data_batch4,data_batch5]
        self.test_batch = [test_batch]
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None
    def set_up_images(self):
        print("Setting Up Training Images and Labels")
        self.training_images = np.vstack(
            [d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        self.training_images = self.training_images.reshape(
            train_len,3,32,32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(
            np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        print("Setting Up Test Images and Labels")
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        self.test_images = self.test_images.reshape(
            test_len,3,32,32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack(
            [d[b"labels"] for d in self.test_batch]), 10)
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size].reshape(batch_size,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
    def test_next_batch(self, batch_size):
        x = self.test_images[self.i:self.i+batch_size].reshape(batch_size,32,32,3)
        y = self.test_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.test_images)
        return x, y
    
ch = CifarHelper()
ch.set_up_images()

## model
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2by2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def global_avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1],
                          strides=[1, 8, 8, 1], padding='SAME')

def norm(x):
    return tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def fc_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,10])
hold_prob1 = tf.placeholder(tf.float32)
hold_prob2 = tf.placeholder(tf.float32)

conv_1 = conv_layer(x,shape=[3,3,3,64])
conv_2 = conv_layer(conv_1,shape=[3,3,64,64])
bn_1 = norm(conv_2)
pooling_1 = max_pool_2by2(bn_1)
dropout_1 = tf.nn.dropout(pooling_1, keep_prob=hold_prob1)

conv_3 = conv_layer(dropout_1,shape=[3,3,64,128])
conv_4 = conv_layer(conv_3,shape=[3,3,128,128])
bn_2 = norm(conv_4)
pooling_2 = avg_pool_2by2(bn_2)
dropout_2 = tf.nn.dropout(pooling_2, keep_prob=hold_prob1)

conv_5 = conv_layer(dropout_2,shape=[3,3,128,256])
conv_6 = conv_layer(conv_5,shape=[3,3,256,256])
bn_3 = norm(conv_6)
pooling_3 = global_avg_pool(bn_3)
flat_1 = tf.reshape(pooling_3,[-1,256])
dropout_3 = tf.nn.dropout(flat_1, keep_prob=hold_prob2)

y_pred = fc_layer(dropout_3,10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()


saver = tf.train.Saver()
Epoch = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500 * Epoch):
        batch = ch.next_batch(100)
        _, loss = sess.run([train, cross_entropy], feed_dict={
            x: batch[0], y_true: batch[1],
            hold_prob1: 0.9, hold_prob2: 0.5})
        print("Epoch:", i//500+1, " ", (i%500+1) * 100, "/ 50000 ", "Loss =", loss)
        
        if i%500 == 499:
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            acc_mean = 0
            for _ in range(100):
                batch = ch.test_next_batch(100)
                acc_mean += sess.run(acc, feed_dict={
                        x:batch[0],y_true:batch[1],
                        hold_prob1: 1.0, hold_prob2: 1.0})
            print("\n")
            print("test acc =", acc_mean / 100)
            print('\n')
    saver.save(sess, './cifar10')



saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './cifar10')
    matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    acc_mean = 0
    for _ in range(100):
        batch = ch.test_next_batch(100)
        acc_mean += sess.run(acc, feed_dict={
                x:batch[0],y_true:batch[1],
                hold_prob1: 1.0, hold_prob2: 1.0})
    print("\n")
    print("test acc =", acc_mean / 100)
    print('\n')
