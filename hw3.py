import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
import model
from skimage import io, transform
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 1
BATCH_SIZE = 10
LOAD_PRETRAIN = True


def load_images(path, mode='id', half=False, pos=None):
    files = os.listdir(path)
    files.sort()
    if mode == 'id':
        return files

    if half and pos == 'front':
        files = files[:int(len(files) / 2)]
    elif half and pos == 'back':
        files = files[int(len(files) / 2):]

    data = []
    for file in files:
        img = io.imread(path + "/" + file)
        img = transform.resize(img, (224, 224), mode="constant")
        img = np.array(img, dtype=float)
        data.append(img)

    data = np.array(data)

    return data


def load_label(path, mode='class'):
    fp = open(path, 'r')
    line = fp.readline()

    label = []
    while line:
        idx = line.find(",")
        label.append((line[:idx], line[idx + 1:-1]))
        line = fp.readline()

    fp.close()
    label = label[1:]
    label.sort()

    for c in range(len(label)):
        label[c] = label[c][1]

    class_type = list(set(label))
    class_type.sort()

    if mode == 'class':
        return class_type

    label = preprocessing.LabelBinarizer().fit_transform(label)  # One-hot Encoding

    return np.array(label)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def train_eval(sess, x_data, y_label, batch_size, train_phase, is_eval, epoch=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    tmp_loss, tmp_acc = 0, 0
    for batch in tqdm(range(n_batch)):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: x_data[start:end], y: y_label[start:end],
                                                       is_training: train_phase})
        tmp_loss += batch_loss * (end - start)
        tmp_acc += batch_acc * (end - start)
    tmp_loss /= n_sample
    tmp_acc /= n_sample
    if train_phase:
        print('\nepoch: {0}, loss: {1:.4f}, acc: {2:.4f}'.format(epoch + 1, tmp_loss, tmp_acc))


def test_eval(sess, x_data, train_phase):
    batch_size = 1
    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    tmp_pred = []
    log = []
    for batch in tqdm(range(n_batch)):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tmp_logits = sess.run(logits, feed_dict={x: x_data[start:end], is_training: train_phase})
        tmp = softmax(np.squeeze(tmp_logits))
        tmp_pred.append(tmp)
    # tmp_pred = np.array(tmp_pred)

    return tmp_pred


def convert_csv(image_id, label, result):
    df = pd.DataFrame(result, index=image_id, columns=label)
    df.to_csv(r"submission.csv")


# data preprocess by yourself
if __name__ == '__main__':
    start_time = time.time()

    train_data = load_images(r"data/train", "n")
    train_label = load_label(r"data/labels.csv", "n")
    # test_data = load_images(r"data/test", "r")

    x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, N_CLASSES), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='train_phase')

    logits = model.VGG16(x=x, is_training=is_training, n_classes=N_CLASSES)

    with tf.name_scope('LossLayer'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1)), tf.float32))

    init = tf.global_variables_initializer()

    restore_variable = [var for var in tf.global_variables() if var.name.startswith('')]

    saver = tf.train.Saver()
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session() as sess:
        if LOAD_PRETRAIN:
            saver.restore(sess, r'model/pretrain/model.ckpt')
        else:
            sess.run(init)

        for i in tqdm(range(EPOCHS)):
            train_eval(sess=sess, x_data=train_data, y_label=train_label,
                       batch_size=BATCH_SIZE, train_phase=True, is_eval=False, epoch=i)
        # saver.save(sess, 'model/model.ckpt')
        del train_data
        del train_label
        ans = test_eval(sess=sess, x_data=load_images(r"data/test", "n", True, 'front'), train_phase=False)
        ans.extend(test_eval(sess=sess, x_data=load_images(r"data/test", "n", True, 'back'), train_phase=False))
    convert_csv(load_images(r"data/test"), load_label(r"data/labels.csv"), np.array(ans))


    end_time = time.time()
    print("\nTime usage:", timedelta(seconds=int(round(end_time - start_time))))
