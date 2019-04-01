import tensorflow as tf
import input_data
import math
import os
import csv
from tqdm import tqdm

# 每层的神经元个数
layer_sizes = [784, 1000, 500, 250, 250, 250, 10] 
# 层数
L = len(layer_sizes) - 1 
num_examples = 60000
# 代数
num_epochs = 40
num_labeled = 60000

starter_learning_rate = 0.02

# epoch after which to begin learning rate decay
decay_after = 15

batch_size = 100
# number of loop iterations
# 每一代都把所有数据循环一遍
num_iter = (num_examples // batch_size) * num_epochs

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
    return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])
# shapes of linear layers，将layer_sizes相邻元素依次两两连接，生成元组，在本例（[784, 1000, 500, 250, 250, 250, 10] ）中生成[(784,1000),(1000,500),(250,250)...(250,10)]
shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))

weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
           'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]}

# scaling factor for noise used in corrupted encoder
noise_std = 0.3

# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]
# join函数把l和u在第一个维度上连接起来
join = lambda l, u: tf.concat([l, u], 0)





# labeled 函数把 x 进行切片,切片起始位置为[0,0],size为[batch_size, -1]
# size 为什么是batch_size 而不是num_labeled
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
# split_lu 函数，切分标签和无标签数据
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)

# 计算均值和方差的滑动平均值，decay为衰减率，用于控制模型的更新速度
ewma = tf.train.ExponentialMovingAverage(decay=0.99)
# this list stores the updates to be made to average mean and variance
# 此列表存储要对平均均值和方差进行的更新
bn_assigns = []


# 归一化
# 参考：https://www.jianshu.com/p/0312e04e4e83
def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]


def update_batch_normalization(batch, l):
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l-1].assign(mean)
    assign_var = running_var[l-1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)

# 编码
def encoder(inputs, noise_std):
    h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    # z:隐藏变量?, m:mean?, v:variance?, h:输出?
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
    for l in range(1, L+1):
        print("Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l])
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
        z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # if training: 
        def training_batch_norm():
            # 训练时的批归一化，有标签和无标签分开进行批归一化
            if noise_std > 0:
                # 加噪声的编码
                # 批归一化后 加噪声 noise
                z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # 未加噪声的编码
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
            return z

        # else: 
        def eval_batch_norm():
            # 评估时用的批归一化函数
            # obtain average mean and variance and use it to normalize the batch
            mean = ewma.average(running_mean[l-1])
            var = ewma.average(running_var[l-1])
            z = batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        # 这是一个根据条件进行流程控制的函数，它有三个参数，pred, true_fn,false_fn ,
        # 它的主要作用是在 pred 为真的时候返回 true_fn 函数的结果，为假的时候返回 false_fn 
        z = tf.cond(training, training_batch_norm, eval_batch_norm)

        if l == L:
            # 输出层使用softmax激活函数
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        else:
            # 隐藏层使用ReLu激活函数,为什么没有gamma？
            h = tf.nn.relu(z + weights["beta"][l-1])
        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print("=== Corrupted Encoder ===")
y_c, corr = encoder(inputs, noise_std)

print("=== Clean Encoder ===")
y, clean = encoder(inputs, 0.0)  # 0.0 -> do not add noise

print("=== Decoder ===")

# 高斯去噪函数
def g_gauss(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est

# 解码
z_est = {}
# 保存各层的去噪代价
d_cost = []
for l in range(L, -1, -1):
    print("Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        u = tf.matmul(z_est[l+1], weights['V'][l])
    u = batch_normalization(u)
    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    z_est_bn = (z_est[l] - m) / tf.sqrt(v + 1-1e-10)
    # 把该层的代价添加到d_cost，reduce_sum(t,1)按行求和
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

# 把各层的去噪代价加起来，计算无监督的总的代价
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
# 监督学习的代价函数
cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))
# 总代价函数
loss = cost + u_cost
# 评估的代价函数
pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))

# 预测正确的样本，返回bool型列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# print(train_step)

# 将批归一化的更新信息添加到train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)
    print("step")

print("===  Loading Data ===")
mnist = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)
# 用来保存模型
saver = tf.train.Saver()

print("===  Starting Session ===")
sess = tf.Session()
# 开始训练的数据位置
i_iter = 0
# 获取最新的一个checkpoint，（如果存在的话）
ckpt = tf.train.get_checkpoint_state('checkpoints/')
if ckpt and ckpt.model_checkpoint_path:
    # 如果checkpoint 存在, 从中恢复参数，并且设置epoch_n 和 i_iter 的值
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = (epoch_n+1) * (num_examples // batch_size)
    print("Restored Epoch ", epoch_n)
else:
    # 如果不存在checkpoint， 如果 checkpoints 目录不存在，则创建该目录。
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    init = tf.global_variables_initializer()
    sess.run(init)

print("=== Training ===")
print("Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")
# tqdm 进度条
# i 是从i_iter 到 num_iter = (num_examples/batch_size) * num_epochs，每num_examples/batch_size为一代
for i in tqdm(range(i_iter, num_iter)):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={inputs: images, outputs: labels, training: True})
    # print(train_step)
    if (i > 1) and ((i+1) % (num_iter // num_epochs) == 0):
        epoch_n = i // (num_examples // batch_size)
        if (epoch_n + 1) >= decay_after:
            # 衰减学习率
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n + 1))  # epoch_n + 1 因为这个学习率是为下一次循环设置的
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
        with open('train_log.csv', 'a') as train_log:
            # 将测试准确率写入 "train_log"文件
            train_log_w = csv.writer(train_log)
            log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
            train_log_w.writerow(log_i)

print("Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

sess.close()
