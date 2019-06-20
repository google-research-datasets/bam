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

import tensorflow as tf
import numpy as np
import PIL.Image
import os
from multiprocessing import dummy as multiprocessing
import saliency

ROW_MIN = 0
COL_MIN = 1
ROW_MAX = 2
COL_MAX = 3

# Same as in https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

SAL_DIR = 'sal_scratch'
ATTR_DIR = 'attr_scratch'
RESNET_SHAPE = (224, 224)
IMG_SHAPE = (128, 128)


def load_imgs(fnames, num_threads=1, shape=RESNET_SHAPE):
  """Load and preprocess images for ResNet by subtracting means."""

  def load_img(f):
    img = PIL.Image.open(tf.gfile.Open(f, 'rb'))
    img = img.convert('RGB').resize(shape, PIL.Image.BILINEAR)
    channel_means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
    img_arr = np.array(img, dtype=np.float32) - channel_means.astype(np.float32)
    return img_arr

  pool = multiprocessing.Pool(num_threads)
  return pool.map(load_img, fnames)


def visualize_pos_attr(image_3d, percentile=99):
  """Returns a 3D tensor as a grayscale 2D tensor by summing the positive attributions of a 3D tensor across axis=2, and then clips values at a given percentile."""

  image_2d = np.sum(image_3d.clip(min=0), axis=2)
  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)
  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def single_attr(map_2d, loc, obj_mask):
  """Given a 2D saliency map, the location of an object, and the binary mask of that object, compute the attribution of the object by averaging over its pixel-wise attributions."""

  obj_mask_resized = np.array(
      PIL.Image.fromarray(obj_mask).resize(
          (loc[COL_MAX] - loc[COL_MIN], loc[ROW_MAX] - loc[ROW_MIN]),
          PIL.Image.BILINEAR))
  avg = np.sum(
      map_2d[:, loc[ROW_MIN]:loc[ROW_MAX], loc[COL_MIN]:loc[COL_MAX]] *
      obj_mask_resized,
      axis=(-1, -2)) / np.count_nonzero(obj_mask_resized)
  return avg


def compute_and_save_attr(model, data, indices, num_threads):
  """Given the name of a model and a set of data, select a set of images based on provided indices, and compute and save their saliency maps and object attributions."""

  base_dir = os.getcwd()
  data_dir = os.path.join(base_dir, 'data', data, 'val')
  model_dir = os.path.join(base_dir, 'models', model)
  sal_output_dir = os.path.join(base_dir, SAL_DIR, model + '-' + data)
  attr_output_dir = os.path.join(base_dir, ATTR_DIR, model + '-' + data)
  if not tf.gfile.Exists(sal_output_dir):
    tf.gfile.MakeDirs(sal_output_dir)
  if not tf.gfile.Exists(attr_output_dir):
    tf.gfile.MakeDirs(attr_output_dir)

  img_names = [sorted(tf.gfile.ListDirectory(data_dir))[i] for i in indices]
  img_paths = [os.path.join(data_dir, img_name) for img_name in img_names]
  imgs = load_imgs(img_paths, num_threads)

  input_name = 'input_tensor:0'
  logit_name = 'resnet_model/final_dense:0'
  conv_name = 'resnet_model/block_layer4:0'

  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], model_dir)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(input_name)
    logit_tensor = graph.get_tensor_by_name(logit_name)
    neuron_selector = tf.placeholder(tf.int32)
    y = logit_tensor[:, neuron_selector]
    pred_tensor = tf.argmax(logit_tensor, 1)

    vg = saliency.GradientSaliency(graph, sess, y, input_tensor)
    gb = saliency.GuidedBackprop(graph, sess, y, input_tensor)
    ig = saliency.IntegratedGradients(graph, sess, y, input_tensor)
    gc = saliency.GradCam(graph, sess, y, input_tensor,
                          graph.get_tensor_by_name(conv_name))

    def single_map(img, img_name):
      pred = sess.run(pred_tensor, feed_dict={input_tensor: [img]})[0]
      vg_mask = vg.GetMask(img, feed_dict={neuron_selector: pred})
      # *s is SmoothGrad
      vgs_mask = vg.GetSmoothedMask(
          img, feed_dict={neuron_selector: pred}, num_threads=50)
      gb_mask = gb.GetMask(img, feed_dict={neuron_selector: pred})
      gbs_mask = gb.GetSmoothedMask(
          img, feed_dict={neuron_selector: pred}, num_threads=50)
      baseline = np.zeros(img.shape) - np.expand_dims(
          np.expand_dims(_CHANNEL_MEANS, 0), 0)
      ig_mask = ig.GetMask(
          img, feed_dict={neuron_selector: pred}, x_baseline=baseline)
      igs_mask = ig.GetSmoothedMask(
          img,
          feed_dict={neuron_selector: pred},
          x_baseline=baseline,
          num_threads=50)
      gc_mask = gc.GetMask(img, feed_dict={neuron_selector: pred})
      gcs_mask = gc.GetSmoothedMask(
          img, feed_dict={neuron_selector: pred}, num_threads=50)
      # gbgc is guided GradCam
      gbgc_mask = gb_mask * gc_mask
      gbgcs_mask = gbs_mask * gcs_mask
      # Also include gradient x input
      masks = np.array([
          vg_mask, vgs_mask, gb_mask, gbs_mask, ig_mask, igs_mask, gc_mask,
          gcs_mask, gbgc_mask, gbgcs_mask, vg_mask * img, vgs_mask * img
      ])
      return masks, pred

    sal_maps = []
    preds = []
    for img, img_name in zip(imgs, img_names):
      sal_path = tf.gfile.Glob(
          os.path.join(sal_output_dir, img_name[:-4] + '*'))
      if len(sal_path) > 0:
        sal_maps.append(np.load(tf.gfile.GFile(sal_path[0], 'rb')))
        preds.append(sal_path[0].split('_')[-1])
        tf.logging.info('Loaded saliency maps for {}.'.format(img_name))
      else:
        masks, pred = single_map(img, img_name)
        sal_maps.append(masks)
        preds.append(pred)
        out_path = os.path.join(sal_output_dir, img_name[:-4] + '_' + str(pred))
        np.save(tf.gfile.GFile(out_path, 'w'), masks)
        tf.logging.info('Saved saliency maps for {}.'.format(img_name))

  # Locate the objects, convert 3D saliency maps to 2D, and compute
  # the attributions of the object segments by averaging over the
  # per-pixel attributions of the objects.
  loc_fpath = os.path.join(base_dir, 'data', data, 'val_loc.txt')
  lines = [tf.gfile.Open(loc_fpath).readlines()[i] for i in indices]
  locs = np.array([[
      int(int(l) * float(RESNET_SHAPE[0]) / IMG_SHAPE[0])
      for l in line.rstrip('\n').split(' ')[-1].split(',')
  ]
                   for line in lines])
  pool = multiprocessing.Pool(num_threads)
  maps_3d = np.array(sal_maps).reshape(-1, RESNET_SHAPE[0], RESNET_SHAPE[1], 3)
  maps_2d = np.array(pool.map(visualize_pos_attr, maps_3d))
  maps_2d = maps_2d.reshape(
      len(indices), int(maps_2d.shape[0] // len(indices)), RESNET_SHAPE[0],
      RESNET_SHAPE[1])
  mask_fpath = os.path.join(base_dir, 'data', data, 'val_mask')
  obj_masks = [
      np.load(tf.gfile.GFile(mask_fpath, 'rb'), allow_pickle=True)[i]
      for i in indices
  ]
  attrs = []
  for i in range(len(indices)):
    attr = single_attr(maps_2d[i], locs[i], obj_masks[i])
    attrs.append(attr)
    out_path = os.path.join(attr_output_dir,
                            img_names[i][:-4] + '_' + str(preds[i]))
    np.save(tf.gfile.GFile(out_path, 'w'), attr)
