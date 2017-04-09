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
BATCH_SIZE = 2
NUM_EPOCHS = 200
S = 7
B = 2
CLASSES = 2
COORD_W = 5
NOOBJ_W = 0.5
PROB_THRESHOLD = 0.25
NMS_THRESHOLD = 0.5
TRAIN_SIZE = 122
alpha = 0.1
EVAL_FREQUENCY = 100
TRAIN_IMG_DIR = '/home/yy/train/'
TRAIN_LABEL_DIR = '/home/yy/labels/'
CLASSES_NAME = ["DaLai","NonDaLai"]
TEST_IMG_DIR = '/home/yy/test1/'
TEST_LABEL_DIR = 'home/yy/labels/'
RES_DIR = '/home/yy/subnets2/'
SAVE_MODEL = '/home/yy/tf_saver_models/model_yolo4.ckpt'
SAVE_TENSORBOARD = '/home/yy/tensorboard'


conv1_weights = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS+2, 64], stddev=0.01, seed=SEED, dtype=tf.float32))
conv1_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01, seed=SEED, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))
conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01, seed=SEED, dtype=tf.float32))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32))
conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.01, seed=SEED, dtype=tf.float32))
conv4_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], stddev=0.01, seed=SEED, dtype=tf.float32))
conv5_biases = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32))

"""separate fc layer to fc1fc2 for coordinate regression and fc3fc4 for classify regression"""
fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE //1024  * 1024, 128], stddev=0.01, seed=SEED, dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([128, S*S*(B*5)], stddev=0.01, seed=SEED, dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[S*S*(B*5)], dtype=tf.float32))

fc3_weights = tf.Variable(tf.truncated_normal([1024, 128], stddev=0.01, seed=SEED, dtype=tf.float32))
fc3_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))
fc4_weights = tf.Variable(tf.truncated_normal([128, S*S*CLASSES], stddev=0.01, seed=SEED, dtype=tf.float32))
fc4_biases = tf.Variable(tf.constant(0.1, shape=[S*S*CLASSES], dtype=tf.float32))

def model(data):
  conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv1_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv2_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv3_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  conv = tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv4_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  conv = tf.nn.conv2d(pool, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, conv5_biases)
  lrelu = tf.maximum(alpha*conv_bias, conv_bias)

  pool = tf.nn.max_pool(lrelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  fc1_shape = pool.get_shape().as_list()
  reshape = tf.reshape(pool, [fc1_shape[0], fc1_shape[1] * fc1_shape[2] * fc1_shape[3]])

  fc1_hidden = tf.matmul(reshape, fc1_weights) + fc1_biases
  fc1_out = tf.maximum(alpha*fc1_hidden, fc1_hidden)

  coors = tf.sigmoid(tf.matmul(fc1_out, fc2_weights) + fc2_biases)

  pool = tf.nn.avg_pool(lrelu, ksize=[1, IMAGE_SIZE*IMAGE_SIZE/1024, IMAGE_SIZE*IMAGE_SIZE/1024, 1], strides=[1, IMAGE_SIZE*IMAGE_SIZE/1024, IMAGE_SIZE*IMAGE_SIZE/1024, 1], padding='SAME')
  
  fc3_shape = pool.get_shape().as_list()
  reshape = tf.reshape(pool, [fc3_shape[0], fc3_shape[1] * fc3_shape[2] * fc3_shape[3]])

  fc3_hidden = tf.matmul(reshape, fc3_weights) + fc3_biases
  fc3_out = tf.maximum(alpha*fc3_hidden, fc3_hidden)

  probs = tf.sigmoid(tf.matmul(fc3_out, fc4_weights) + fc4_biases)
  output = []

  for i in range(BATCH_SIZE):
    for j in range(S*S):
      for k in range(10):
        output.append(coors[i,j*B*5+k])
      for k in range(CLASSES):
        output.append(probs[i,j*CLASSES+k])
  output = tf.reshape(output, [BATCH_SIZE, S*S*(B*5+CLASSES)])

  return output


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
      cv2.putText(img, CLASSES_NAME[classes[i]] + ' : %.2f' % results[i][4], (x1+5,y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

  cv2.imwrite(RES_DIR + img_path.split('/')[-1], img)

def get_next_minibatch(offset, path_list):
  if offset+BATCH_SIZE > len(path_list):
    random.shuffle(path_list)
    return path_list[:BATCH_SIZE]
  else:
    return path_list[offset:offset+BATCH_SIZE]

def extract_data_yolo(path_list, train=True):
  if train:
    data = numpy.ndarray(shape=(len(path_list),IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS + 2),dtype=numpy.float32)

    """add original position information"""
    for i in range(len(path_list)):
      for j in range(IMAGE_SIZE):
          data[i,j,:,-2] = j

    for i in range(len(path_list)):
      for j in range(IMAGE_SIZE):
          data[i,:,j,-1] = j

    for i in range(len(path_list)):
      img = Image.open(TRAIN_IMG_DIR+path_list[i]+'.jpg')
      img_resize = img.resize((IMAGE_SIZE,IMAGE_SIZE))
      data[i,:,:,:-2] = numpy.array(img_resize).astype(numpy.float32).reshape(IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
    
    return data
  else:
    data = numpy.ndarray(shape=(1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS + 2), dtype=numpy.float32)

    for i in range(IMAGE_SIZE):
        data[0,i,:,-2] = i

    for i in range(IMAGE_SIZE):
        data[0,:,i,-1] = i

    img = Image.open(path_list)
    img_resize = img.resize((IMAGE_SIZE,IMAGE_SIZE))
    data[0,:,:,:-2] = numpy.array(img_resize).astype(numpy.float32).reshape(1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
    return data

def iou(box1,box2):
  tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
  lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
  if tb < 0 or lr < 0 : intersection = 0
  else : intersection =  tb*lr
  return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


def extract_labels_yolo(path_list, train=True):
  if train:
    root = TRAIN_LABEL_DIR
  else:
    root = TEST_LABEL_DIR
  labels = numpy.ndarray(shape=(len(path_list),S*S*(B*5+CLASSES)), dtype=numpy.float32)
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      if j%(B*5+CLASSES) == 0 or j%(B*5+CLASSES) == 5:
        labels[i][j] = 1.00001
      else:
        labels[i][j] = 0
  for i in range(len(path_list)):
    with open(root + path_list[i] + '.txt',"r") as f:
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

def loss_func_yolo(output, label):
  res = 0

  for i in range(BATCH_SIZE):
    for j in range(0, S*S*(B*5+CLASSES), B*5+CLASSES):
      highest_bbox = output[i][j+4]-output[i][j+9]
      """here we only compute the loss of bbox which have the highest confidence"""
      """we use tf.sign(tf.maximum(highest_bbox,0)) to do that"""

      res += COORD_W * tf.sign(tf.maximum(highest_bbox,0)) * tf.sign(label[i][j+2]) * (
                                                             tf.square(output[i][j] - label[i][j]) + 
                                                             tf.square(output[i][j+1]-label[i][j+1]) + 
                                                             tf.square(output[i][j+2]/(label[i][j+2]+1e-7) - 1) + 
                                                             tf.square(output[i][j+3]/(label[i][j+3]+1e-7) - 1))

      res += tf.sign(tf.maximum(highest_bbox,0)) * tf.sign(label[i][j+2]) * (tf.square(output[i][j+4] - label[i][j+4]))

      res += NOOBJ_W * tf.sign(tf.maximum(highest_bbox,0)) * tf.sign(tf.floor(label[i][j])) * (tf.square(output[i][j+4] - label[i][j+4]))

      res += COORD_W * tf.sign(tf.maximum(-highest_bbox,0)) * tf.sign(label[i][j+7]) * (
                                                              tf.square(output[i][j+5] - label[i][j+5]) + 
                                                              tf.square(output[i][j+6]-label[i][j+6]) + 
                                                              tf.square(output[i][j+7]/(label[i][j+7]+1e-7) - 1) + 
                                                              tf.square(output[i][j+8]/(label[i][j+8]+1e-7) - 1))

      res += tf.sign(tf.maximum(-highest_bbox,0)) * tf.sign(label[i][j+7]) * (tf.square(output[i][j+9] - label[i][j+9]))

      res += NOOBJ_W * tf.sign(tf.maximum(-highest_bbox,0)) * tf.sign(tf.floor(label[i][j+5])) * (tf.square(output[i][j+9] - label[i][j+9]))

      res += tf.sign(label[i][j+7]) * (tf.square(output[i][j+10] - label[i][j+10]) + tf.square(output[i][j+11] - label[i][j+11]))

  return res/BATCH_SIZE

# def loss_func_yolo(output, label):
#   res = 0

#   for i in range(BATCH_SIZE):
#     for j in range(0, S*S*(B*5+CLASSES), B*5+CLASSES):
#       res += COORD_W * tf.sign(label[i][j+2]) * (tf.square(output[i][j] - label[i][j]) + tf.square(output[i][j+1]-label[i][j+1]) + 
#                                                tf.square(output[i][j+2]/(label[i][j+2]+1e-7) - 1) + 
#                                                tf.square(output[i][j+3]/(label[i][j+3]+1e-7) - 1))

#       res += tf.sign(label[i][j+2]) * (tf.square(output[i][j+4] - label[i][j+4]))

#       res += NOOBJ_W * tf.sign(tf.floor(label[i][j])) * (tf.square(output[i][j+4] - label[i][j+4]))

#       res += COORD_W * tf.sign(label[i][j+7]) * (tf.square(output[i][j+5] - label[i][j+5]) + tf.square(output[i][j+6]-label[i][j+6]) + 
#                                                tf.square(output[i][j+7]/(label[i][j+7]+1e-7) - 1) + 
#                                                tf.square(output[i][j+8]/(label[i][j+8]+1e-7) - 1))

#       res += tf.sign(label[i][j+7]) * (tf.square(output[i][j+9] - label[i][j+9]))

#       res += NOOBJ_W * tf.sign(tf.floor(label[i][j+5])) * (tf.square(output[i][j+9] - label[i][j+9]))

#       res += tf.sign(label[i][j+7]) * (tf.square(output[i][j+10] - label[i][j+10]) + tf.square(output[i][j+11] - label[i][j+11]))

#   return res

def test_from_img(img, test_model, display_loss=False):
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, test_model)
    data = extract_data_yolo(img, train=False)
    out = sess.run(model(data))
    if display_loss:
      label = extract_labels_yolo([img], train=False)
      print('loss: %.6f' % loss_func_yolo(out, label))
    results,classes = get_results(out)
    show_results(img, results, classes)

def test_from_dir(imgdir, test_model, display_loss=False):
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, test_model)
    if display_loss:
      loss = 0
      for root, dirs, files in os.walk(imgdir[:-1]):
        for file in files:
          img = os.path.join(root, file)
          label = extract_labels_yolo([img], train=False)
          data = extract_data_yolo(img, train=False)
          out = sess.run(model(data))
          loss += loss_func_yolo(out, label)
          results,classes = get_results(out)
          show_results(img, results, classes)
      print('loss: %.6f' % loss)
    else:
      for root, dirs, files in os.walk(imgdir[:-1]):
        for file in files:
          img = os.path.join(root, file)
          data = extract_data_yolo(img, train=False)
          out = sess.run(model(data))
          results,classes = get_results(out)
          show_results(img, results, classes)

def preprocessing(imgs):
  res = []
  for i in range(BATCH_SIZE):
    res.append(tf.image.per_image_standardization(imgs[i]))
  return tf.stack(res)

def main(argv=None):
  num_epochs = NUM_EPOCHS
  train_img_list = []
  for rt,dirs,filenames in os.walk(TRAIN_IMG_DIR):
    for filename in filenames:
      train_img_list.append(filename[:-4])

  numpy.random.shuffle(train_img_list)
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS+2))
  train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, S*S*(B*5+CLASSES)))

  train_data_node = preprocessing(train_data_node)
  logits = model(train_data_node)
  loss = loss_func_yolo(logits, train_labels_node)

  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                  tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases) +
                  tf.nn.l2_loss(fc4_weights) + tf.nn.l2_loss(fc4_biases))

  loss += 5e-4 * regularizers

  batch = tf.Variable(0, dtype=tf.float32)

  learning_rate = tf.train.exponential_decay(
      0.001,                
      batch * BATCH_SIZE,  
      10000,          
      0.95,
      staircase=True)

  optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=batch)

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
        #print('weight: %.5f' % sess.run(conv1_weights)[0,0,0,0])
        writer.add_summary(summary, step)
    save_path = saver.save(sess, SAVE_MODEL)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO demo')
    parser.add_argument('--train', help='train the model', action='store_true')
    parser.add_argument('--test', help='test the model', action='store_true')
    parser.add_argument('--test_img_path', help='img path to test', type=str)
    parser.add_argument('--display_loss', default=False, help='whether display the loss', action='store_true')
    parser.add_argument('--test_model', help='model to test', type=str)
    args = parser.parse_args()

    return args
if __name__ == '__main__':
  args = parse_args()
  if args.train and args.test:
    print('Error: cannot train and test at the same time')
  elif args.train:
    tf.app.run()
  elif args.test_img_path[-1] == '/':
    test_from_dir(args.test_img_path, args.test_model, args.display_loss)
  else:
    test_from_img(args.test_img_path, args.test_model, args.display_loss)
