from Model import build_model
import tensorflow as tf
import os
from keras.utils import *
from testing import *
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
def IOU(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def train(epochs, num_steps):
    dir_images = "Dataset/JPEGImages/480p"
    dir_annotations = "Dataset/Annotations/480p"

    classes, classidx = find_classes(dir_images)
    datas = make_dataset(dir_images, dir_annotations)
    x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='X') #Need to define the shape of x
    y = tf.placeholder(tf.float32, shape=[None, 256,256, 1], name='Y')
    logits = build_model(x, 0.5, 128)
    model = tf.identity(logits, name='logits')
    Cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    # Initialize all variables
    sess.run(init)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # variables need to be initialized before any sess.run() calls
        tf.global_variables_initializer().run()

        for X_batch, y_batch in generator(datas,32):
            feed_dict = {x: X_batch, y: y_batch}
            sess.run(optimizer, feed_dict)
'''
def prediction():
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    # Initialize all variables
    sess.run(init)
    test_accuracy = tf.metrics.accuracy()
    #model = saver.restore(sess, "model.ckpt")
    #logits = model(x_test)
    #prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction,y_test)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
'''
if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    epochs = 1
    num_steps = 500
    batch_size = 128
    display_step = 100
    keep_probability = 0.7
    learning_rate = 0.001
    display_freq = 100

    train(epochs,num_steps)
