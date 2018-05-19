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
plt.switch_backend('agg')

INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 50
BATCH_SIZE = 10
LOAD_PRETRAIN = True
EPOCHS_LIST = []
LOSS_LIST = []
ACC_LIST = []


def image_gen(path, start, end):
    files = os.listdir(path)
    files.sort()

    files = files[start:end]
    data = []
    for file in files:
        img = io.imread(path + "/" + file)
        img = transform.resize(img, (224, 224), mode="constant")
        img = np.array(img, dtype=float)
        data.append(img)

    data = np.array(data)

    return data


def load_images_id(path):
    files = os.listdir(path)
    files.sort()
    return files


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
    n_sample = len(os.listdir(x_data))
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    tmp_loss, tmp_acc = 0, 0
    for batch in tqdm(range(n_batch)):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        train_data = image_gen(x_data, start, end)
        _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: train_data, y: y_label[start:end],
                                                       is_training: train_phase})
        tmp_loss += batch_loss * (end - start)
        tmp_acc += batch_acc * (end - start)
    tmp_loss /= n_sample
    tmp_acc /= n_sample
    if train_phase:
        print('\nepoch: {0}, loss: {1:.4f}, acc: {2:.4f}'.format(epoch + 1, tmp_loss, tmp_acc))
        EPOCHS_LIST.append(epoch + 1), LOSS_LIST.append(tmp_loss), ACC_LIST.append(tmp_acc)


def test_eval(sess, x_data, train_phase):
    batch_size = 1
    n_sample = len(os.listdir(x_data))
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    tmp_pred = []
    log = []
    for batch in tqdm(range(n_batch)):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        test_data = image_gen(x_data, start, end)
        tmp_logits = sess.run(logits, feed_dict={x: test_data, is_training: train_phase})
        tmp = softmax(np.squeeze(tmp_logits))
        tmp_pred.append(tmp)
        tmp_pred = np.array(tmp_pred)

    return tmp_pred


def convert_csv(image_id, label, result):
    df = pd.DataFrame(result, index=image_id, columns=label)
    df.to_csv(r"submission.csv")


def plot(x, y, type):
    plt.plot(x, y, label=type)
    plt.ylabel(type)
    plt.xlabel('epoch')
    plt.legend([type], loc='upper left')
    plt.savefig(type)
    plt.clf()


if __name__ == '__main__':
    start_time = time.time()

    train_label = load_label(r"data/labels.csv", "n")

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
    with tf.Session() as sess:
        if LOAD_PRETRAIN:
            saver.restore(sess, r'model/pretrain/model.ckpt')
        else:
            sess.run(init)

        for i in tqdm(range(EPOCHS)):
            train_eval(sess=sess, x_data=r"data/train", y_label=train_label,
                       batch_size=BATCH_SIZE, train_phase=True, is_eval=False, epoch=i)
        # saver.save(sess, 'model/model.ckpt')
        ans = test_eval(sess=sess, x_data=r"data/test", train_phase=False)
    convert_csv(load_images_id(r"data/test"), load_label(r"data/labels.csv"), ans)
    plot(EPOCHS_LIST, LOSS_LIST, 'loss')
    plot(EPOCHS_LIST, ACC_LIST, 'acc')

    end_time = time.time()
    print("\nTime usage:", timedelta(seconds=int(round(end_time - start_time))))
