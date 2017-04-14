from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import cv2
import numpy
from PIL import Image
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 224
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
SEED = 66478
BATCH_SIZE = 1
NUM_EPOCHS = 20
S = 7
B = 2
CLASSES = 2
COORD_W = 5
NOOBJ_W = 0.5
PROB_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
TRAIN_SIZE = 122
alpha = 0.1
EVAL_FREQUENCY = 100
TRAIN_IMG_DIR = '/home/yy/train/'
TRAIN_LABEL_DIR = '/home/yy/labels/'
CLASSES_NAME = ["DaLai","NonDaLai"]
TEST_IMG_PATH = '/home/yy/109.jpg'
RES_DIR = '/home/yy/pred_shuffle/'
SAVE_MODEL = '/home/yy/tf_saver_models/model_newls.ckpt'
SAVE_TENSORBOARD = '/home/yy/tensorboard'
TEST_MODEL = '/home/yy/tf_saver_models/model_newls.ckpt'


conv1_weights = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64], stddev=0.01, seed=SEED, dtype=tf.float32))
conv1_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01, seed=SEED, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))


fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE //16  * 128, 128], stddev=0.01, seed=SEED, dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([128, S*S*(B*5+CLASSES)], stddev=0.01, seed=SEED, dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[S*S*(B*5+CLASSES)], dtype=tf.float32))

def model(data, train=False):
  conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv1_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv2_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  fc1_shape = pool.get_shape().as_list()
  reshape = tf.reshape(pool, [fc1_shape[0], fc1_shape[1] * fc1_shape[2] * fc1_shape[3]])

  fc1_hidden = tf.matmul(reshape, fc1_weights) + fc1_biases
  fc1_out = tf.maximum(alpha*fc1_hidden, fc1_hidden)

  return tf.matmul(fc1_out, fc2_weights) + fc2_biases

def nms(dets, thresh):
  """Non maximum suppression"""
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
      i = order[0]
      keep.append(i)
      xx1 = numpy.maximum(x1[i], x1[order[1:]])
      yy1 = numpy.maximum(y1[i], y1[order[1:]])
      xx2 = numpy.minimum(x2[i], x2[order[1:]])
      yy2 = numpy.minimum(y2[i], y2[order[1:]])

      w = numpy.maximum(0.0, xx2 - xx1 + 1)
      h = numpy.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h
      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      inds = numpy.where(ovr <= thresh)[0]
      order = order[inds + 1]

  return keep

def get_results(output):
  results = []
  classes = []
  probs = numpy.ndarray(shape=[CLASSES,])
  for p in range(B):
    for j in range(4 + p*5, S*S*(B*5+CLASSES), B*5+CLASSES):
      for i in range(CLASSES):
        probs[i] = output[0][j] * output[0][j + 1+ (B-1-p)*5 + i]

      cls_ind = probs.argsort()[::-1][0]
      if probs[cls_ind] > PROB_THRESHOLD:
        results.append([output[0][j-4] - output[0][j-2]/2, output[0][j-3] - output[0][j-3]/2, output[0][j-4] + output[0][j-2]/2, output[0][j-3] + output[0][j-3]/2, probs[cls_ind]])
        classes.append(cls_ind)

  res = numpy.array(results).astype(numpy.float32)
  if len(res) != 0:
    keep = nms(res, NMS_THRESHOLD)
    results_ = []
    classes_ = []
    for i in keep:
      results_.append(results[i])
      classes_.append(classes[i])

    return results_,classes_
  else:
    return [],[]

def show_results(img_path, results, classes):
  img = cv2.imread(img_path).copy()
  if len(results) != 0:
    for i in range(len(results)):
      x1 = int(results[i][0]*img.shape[1])
      y1 = int(results[i][1]*img.shape[0])
      x2 = int(results[i][2]*img.shape[1])
      y2 = int(results[i][3]*img.shape[0])
      score = results[i][4]
      cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
      cv2.putText(img, CLASSES_NAME[classes[i]] + ' : %.2f' % results[i][4], (x1+5,y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

  cv2.imwrite(RES_DIR + img_path.split('/')[-1], img)

def get_next_minibatch(offset, path_list):
  if offset+BATCH_SIZE > len(path_list):
    # random.shuffle(path_list)
    return path_list[:BATCH_SIZE]
  else:
    return path_list[offset:offset+BATCH_SIZE]

def extract_data_yolo(path_list, train=True):
  if train:
    data = numpy.ndarray(shape=(len(path_list),IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),dtype=numpy.float32)

    for i in range(len(path_list)):
      img = Image.open(TRAIN_IMG_DIR+path_list[i]+'.jpg')
      img_resize = img.resize((IMAGE_SIZE,IMAGE_SIZE))
      data[i] = numpy.array(img_resize).astype(numpy.float32).reshape(IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)

    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    return data
  else:
    data = numpy.ndarray(shape=(1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS), dtype=numpy.float32)
    img = Image.open(path_list)
    img_resize = img.resize((IMAGE_SIZE,IMAGE_SIZE))
    data = numpy.array(img_resize).astype(numpy.float32).reshape(1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    return data

def iou(box1,box2):
  tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
  lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
  if tb < 0 or lr < 0 : intersection = 0
  else : intersection =  tb*lr
  return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


def extract_labels_yolo(path_list):
  labels = numpy.ndarray(shape=(len(path_list),S*S*(B*5+CLASSES)), dtype=numpy.float32)
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      if j%(B*5+CLASSES) == 0 or j%(B*5+CLASSES) == 5:
        labels[i][j] = 1.00001
      else:
        labels[i][j] = 0
  for i in range(len(path_list)):
    with open(TRAIN_LABEL_DIR + path_list[i] + '.txt',"r") as f:
      lines = f.readlines()
      for j in range(len(lines)):
        data = lines[j].split()
        col_no = int(float(data[1])*IMAGE_SIZE/(IMAGE_SIZE/S)+1)
        row_no = int(float(data[2])*IMAGE_SIZE/(IMAGE_SIZE/S)+1)
        grid_no = (row_no-1)*S+col_no
        # labels[i,(B*5+CLASSES)*grid_no-1] = float(data[0])
        labels[i,(B*5+CLASSES)*grid_no-CLASSES + int(data[0])] = 1
        for k in range(B):
          labels[i,(B*5+CLASSES)*(grid_no-1) + 5*k] = float(data[1])
          labels[i,(B*5+CLASSES)*(grid_no-1) + 5*k + 1] = float(data[2])
          labels[i,(B*5+CLASSES)*(grid_no-1) + 5*k + 2] = float(data[3])
          labels[i,(B*5+CLASSES)*(grid_no-1) + 5*k + 3] = float(data[4])
          labels[i,(B*5+CLASSES)*(grid_no-1) + 5*k + 4] = 1

  return labels

def loss_func_yolo(output, exp):
  res = 0

  for i in range(BATCH_SIZE):
    for j in range(0, S*S*(B*5+CLASSES), B*5+CLASSES):
      res += COORD_W * tf.sign(exp[i][j+2]) * (tf.square(output[i][j] - exp[i][j]) + tf.square(output[i][j+1]-exp[i][j+1]) + 
                                               tf.square(tf.sqrt(tf.abs(output[i][j+2])) - tf.sqrt(exp[i][j+2])) + 
                                               tf.square(tf.sqrt(tf.abs(output[i][j+3])) - tf.sqrt(exp[i][j+3])))

      res += tf.sign(exp[i][j+2]) * (tf.square(output[i][j+4] - exp[i][j+4]))

      res += NOOBJ_W * tf.sign(tf.floor(exp[i][j])) * (tf.square(output[i][j+4] - exp[i][j+4]))

      res += COORD_W * tf.sign(exp[i][j+7]) * (tf.square(output[i][j+5] - exp[i][j+5]) + tf.square(output[i][j+6]-exp[i][j+6]) + 
                                               tf.square(tf.sqrt(tf.abs(output[i][j+7])) - tf.sqrt(exp[i][j+7])) + 
                                               tf.square(tf.sqrt(tf.abs(output[i][j+8])) - tf.sqrt(exp[i][j+8])))

      res += tf.sign(exp[i][j+7]) * (tf.square(output[i][j+9] - exp[i][j+9]))

      res += NOOBJ_W * tf.sign(tf.floor(exp[i][j+5])) * (tf.square(output[i][j+9] - exp[i][j+9]))

      res += tf.sign(exp[i][j+7]) * (tf.square(output[i][j+10] - exp[i][j+10]) + tf.square(output[i][j+11] - exp[i][j+11]))

  return res

def test(img):
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, TEST_MODEL)
    data = extract_data_yolo(img, False)
    out = sess.run(model(data))
    results,classes = get_results(out)
    show_results(img, results, classes)

def main(argv=None):
  num_epochs = NUM_EPOCHS
  train_img_list = []
  for rt,dirs,filenames in os.walk(TRAIN_IMG_DIR):
    for filename in filenames:
      train_img_list.append(filename[:-4])

  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, S*S*(B*5+CLASSES)))

  logits = model(train_data_node, True)
  loss = loss_func_yolo(logits, train_labels_node)

  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

  loss += 5e-4 * regularizers

  batch = tf.Variable(0, dtype=tf.float32)

  learning_rate = tf.train.exponential_decay(
      0.01,                
      batch * BATCH_SIZE,  
      TRAIN_SIZE,          
      0.95,
      staircase=True)

  op_func = tf.train.MomentumOptimizer(learning_rate,0.9)

  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 0.5)
  optimizer = op_func.apply_gradients(zip(grads, tvars), global_step=batch)

  tf.summary.scalar("loss", loss)
  tf.summary.scalar("lr", learning_rate)
  merged_summary = tf.summary.merge_all()
  with tf.Session() as sess:

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    print('Initialized!')
    writer = tf.summary.FileWriter(SAVE_TENSORBOARD, sess.graph)

    for step in xrange(int(num_epochs * TRAIN_SIZE) // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
      batch_data = extract_data_yolo(get_next_minibatch(offset, train_img_list))
      batch_labels = extract_labels_yolo(get_next_minibatch(offset, train_img_list))

      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}

      _,los,summary = sess.run([optimizer, loss, merged_summary], feed_dict=feed_dict)

      if step % EVAL_FREQUENCY == 0:
        print('loss: %.6f' % los)
        writer.add_summary(summary, step)
    save_path = saver.save(sess, SAVE_MODEL)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO demo')
    parser.add_argument('--train', help='train the model', action='store_true')
    parser.add_argument('--test', help='test the model', action='store_true')
    parser.add_argument('--test_img_path', help='img path to test', type=str)

    args = parser.parse_args()

    return args
if __name__ == '__main__':
  args = parse_args()
  if args.train and args.test:
    print('Error: cannot train and test at the same time')
  elif args.train:
    tf.app.run()
  else:
    test(args.test_img_path)
