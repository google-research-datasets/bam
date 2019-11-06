"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import tensorflow as tf
import numpy as np
import random
from multiprocessing import dummy as multiprocessing
from absl import app
from absl import flags
from attributions import compute_and_save_attr

flags.DEFINE_integer('num_imgs', 10,
                     'Number of images to compute metrics over.')
flags.DEFINE_integer(
    'num_threads', 50,
    'Number of threads to use when perform images operations.')
flags.DEFINE_integer(
    'seed', 0, 'The seed to use when randomly sample from a set of inputs.'
    'Make sure sal_scratch is empty before changing the seed.')
flags.DEFINE_boolean(
    'scratch', True,
    'Compute metrics from scratch for num_imgs example inputs.')
flags.DEFINE_list('metrics', ['MCS', 'IDR', 'IIR'],
                  'List of metrics to evaluate.')
FLAGS = flags.FLAGS

ATTR_DIR = ['attr_reprod', 'attr_scratch']
METHOD_NAMES = [
    'Vanilla Gradient', 'Gradient SmoothGrad', 'Guided Backprop',
    'Guided Backprop SmoothGrad', 'Integrated Gradient',
    'Integrated Gradient SmoothGrad', 'GradCam', 'GradCam SmoothGrad',
    'Guided GradCam', 'Guided GradCam SmoothGrad', 'Gradient x Input',
    'Gradient SmoothGrad x Input'
]

# Index into METHOD_NAMES for the desired method names. Ordered roughly
# in the best to worst performance under the BIM metric.
METHOD_INDICES = [6, 0, 1, 4, 5, 10, 2, 8]

# For MCS, IDR, and IIR, the configs correspond to the names of the model-data
# pair (e.g., model1, data1, model2, data2). For RMCS, the config corresponds
# to the prefix of the model name and the data name.
METRIC_CONFIG = {
    'MCS': ['obj', 'obj', 'scene', 'scene'],
    'IDR': ['scene', 'scene_only', 'scene', 'scene'],
    'IIR': ['scene_only', 'bamboo_forest', 'scene_only', 'bamboo_forest_patch'],
    'RMCS': ['scene', 'dog_bedroom']
}

# BIM models have 10 classes so there are 10 models for RMC testing.
NUM_RELATIVE_MODELS = 10

# BIM's dataset for IIR contains 100 images so sample up to 100.
MAX_SAMPLE_INDEX = 100

# MCS and IDR are evaluated on 10000 images, IIR and RMCS are on 100.
NUM_TOTAL_IMAGES = {
    'MCS': 10000,
    'IDR': 10000,
    'IIR': 100,
    'RMCS': 100,
}

BASE_DIR = os.getcwd()


def get_global_indices(metric):
  """When computing from scratch, generate random global image indices from a fixed seed.

  When reproducing results given attributions, enumerate all indices of
  NUM_TOTAL_IMAGES['metric'].
  """

  if not FLAGS.scratch:
    return range(NUM_TOTAL_IMAGES[metric])
  random.seed(FLAGS.seed)
  return sorted(random.sample(range(NUM_TOTAL_IMAGES[metric]), FLAGS.num_imgs))


def corr_indices(model, data, metric):
  """Given the name of a model and a set of data, return the indices of the images that are correctly classified by the model."""

  label_fpath = os.path.join(BASE_DIR, 'data', data, 'val.txt')
  labels = [
      int(l.split(' ')[-1]) for l in tf.gfile.Open(label_fpath).readlines()
  ]
  img_fnames = tf.gfile.ListDirectory(
      os.path.join(BASE_DIR, ATTR_DIR[FLAGS.scratch], model + '-' + data))
  preds = [int(p.split('_')[-1]) for p in sorted(img_fnames)]
  attr_indices = range(len(preds))
  if FLAGS.scratch:
    attr_indices = range(FLAGS.num_imgs)
  global_indices = get_global_indices(metric)
  corr = [i for i in attr_indices if preds[i] == labels[global_indices[i]]]
  return corr


def load_pos_neg_attr(model_pos, data_pos, model_neg, data_neg):
  """Load two sets of attributions from model_pos-data_pos and model_neg-data_neg.

  Filter and only return attributions of correctly classified inputs.

  Args:
      model_pos: the model name for which objects have positive attributions.
      data_pos: the data name for which objects have positive attributions.
      model_neg: the model name for which objects have negative attributions.
      data_neg: the data name for which objects have negative attributions.

  Returns:
      arrays of attributions corresponding to model_pos-data_pos and
      model_neg-data_neg pairs.
  """

  dir_pos = os.path.join(BASE_DIR, ATTR_DIR[FLAGS.scratch],
                         model_pos + '-' + data_pos)
  pool = multiprocessing.Pool(FLAGS.num_threads)
  attr_pos = np.array(
      pool.map(
          lambda f: np.load(tf.gfile.GFile(os.path.join(dir_pos, f), 'rb')),
          sorted(tf.gfile.ListDirectory(dir_pos))))
  dir_neg = os.path.join(BASE_DIR, ATTR_DIR[FLAGS.scratch],
                         model_neg + '-' + data_neg)
  attr_neg = np.array(
      pool.map(
          lambda f: np.load(tf.gfile.GFile(os.path.join(dir_neg, f), 'rb')),
          sorted(tf.gfile.ListDirectory(dir_neg))))
  if FLAGS.scratch:
    attr_pos = attr_pos[:FLAGS.num_imgs]
    attr_neg = attr_neg[:FLAGS.num_imgs]
  return attr_pos, attr_neg


def MCS(model_pos, data_pos, model_neg, data_neg, relative=False):
  """Compute the model contrast score defined as the average attribution

  difference between model_pos-data_pos and model_neg-data_neg.

  Args:
      model_pos: the model name for which objects have positive attributions.
      data_pos: the data name for which objects have positive attributions.
      model_neg: the model name for which objects have negative attributions.
      data_neg: the data name for which objects have negative attributions.
  """

  metric = 'RMCS' if relative else 'MCS'
  attr_pos, attr_neg = load_pos_neg_attr(model_pos, data_pos, model_neg,
                                         data_neg)
  corr_pos = corr_indices(model_pos, data_pos, metric)
  corr_neg = corr_indices(model_neg, data_neg, metric)
  corr = [i for i in corr_pos if i in corr_neg]
  for j in METHOD_INDICES:
    print(','.join(
        map(str, [METHOD_NAMES[j],
                  np.mean((attr_pos - attr_neg)[corr, j])])))


def IDR(model, data_pos, data_neg):
  """Compute the input dependence rate defined as the percentage of examples where data_pos is attributed higher than data_neg.

  Args:
      model: name of the model being evaluated.
      data_pos: the data name for positive attributions.
      data_neg: the data name for negative attributions.
  """

  attr_pos, attr_neg = load_pos_neg_attr(model, data_pos, model, data_neg)
  corr_pos = corr_indices(model, data_pos, 'IDR')
  corr_neg = corr_indices(model, data_neg, 'IDR')
  corr = [i for i in corr_pos if i in corr_neg]
  for j in METHOD_INDICES:
    count = sum(d > 0 for d in (attr_pos - attr_neg)[corr, j])
    print(','.join(map(str, [METHOD_NAMES[j], count / float(len(corr))])))


def IIR(model, data, data_patch, threashold=0.1):
  """Compute the input independence rate defined as the percentage of examples where the difference between data and data_patch is less than threshold.

  Args:
      model: name of the model being evaluated.
      data: name of the data directory that contains scene-only images.
      data_patch: name of the data directory that contains functionally
        insignificant object patches.
  """

  attr, attr_patch = load_pos_neg_attr(model, data, model, data_patch)
  corr = corr_indices(model, data, 'IIR')
  corr_patch = corr_indices(model, data_patch, 'IIR')
  corr = [i for i in corr if i in corr_patch]
  for j in METHOD_INDICES:
    diff = abs(attr[corr, j] - attr_patch[corr, j])
    count = sum(diff < threashold * attr[corr, j])
    print(','.join(map(str, [METHOD_NAMES[j], count / float(len(corr))])))


def main(argv):
  for metric in FLAGS.metrics:
    print('Results for {}:'.format(metric))

    global_indices = get_global_indices(metric)
    if metric == 'RMCS':
      model_prefix, data = METRIC_CONFIG[metric]
      if FLAGS.scratch:
        for i in range(1, NUM_RELATIVE_MODELS + 1):
          compute_and_save_attr(model_prefix + str(i), data, global_indices,
                                FLAGS.num_threads)
      for i in range(1, NUM_RELATIVE_MODELS):
        print('MCS between', model_prefix + str(i), 'and',
              model_prefix + str(NUM_RELATIVE_MODELS))
        MCS(model_prefix + str(i), data,
            model_prefix + str(NUM_RELATIVE_MODELS), data, relative=True)
    else:
      model1, data1, model2, data2 = METRIC_CONFIG[metric]
      if FLAGS.scratch:
        compute_and_save_attr(model1, data1, global_indices, FLAGS.num_threads)
        compute_and_save_attr(model2, data2, global_indices, FLAGS.num_threads)
      if metric == 'MCS':
        MCS(model1, data1, model2, data2)
      if metric == 'IDR':
        IDR(model1, data1, data2)
      if metric == 'IIR':
        IIR(model1, data1, data2)


if __name__ == '__main__':
  app.run(main)
