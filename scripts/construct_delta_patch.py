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
tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
from PIL import Image
import os
from multiprocessing import dummy as multiprocessing
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_bool

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def load_imgs(fnames, shape=(224, 224)):
  pool = multiprocessing.Pool(100)
  return pool.map(
      lambda f: np.array(
          Image.open(tf.gfile.GFile(f)).convert('RGB').resize(
              shape, Image.BILINEAR)).astype(np.float32) - np.expand_dims(
                  np.expand_dims(_CHANNEL_MEANS, 0), 0).astype(np.float32),
      fnames)


def show_imgs(img, adv_img, title=''):
  means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
  img = (img + means).astype(np.uint8)
  adv_img = (adv_img + means).astype(np.uint8)

  plt.subplot(1, 3, 1)
  plt.axis('off')
  plt.imshow(img)
  plt.title('org_img')

  plt.subplot(1, 3, 2)
  plt.axis('off')
  plt.imshow(adv_img)
  plt.title('patch_img')

  plt.subplot(1, 3, 3)
  diff_img = img - adv_img
  plt.imshow(np.abs(1.0 / np.amax(diff_img)) * diff_img)
  plt.title('diff')
  plt.axis('off')
  plt.show()


def save_img(img, adv_img, fname):
  means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
  img = (img + means).astype(np.uint8)
  adv_img = (adv_img + means).astype(np.uint8)

  pil_img = Image.fromarray(adv_img, 'RGB')
  pil_img.save(
      tf.gfile.GFile(
          os.path.join(os.getcwd(), 'data', 'bamboo_forest_patch',
                       fname + '.jpg'), 'w'))


def build_graph(sess,
                num_steps,
                step_size,
                dog_imgs,
                loc,
                dog_mask,
                func_name='resnet_model/final_dense:0',
                input_name='input_tensor:0',
                model_name='scene_only',
                image_bounds=[0 - _B_MEAN, 255 - _R_MEAN],
                alpha=0.01,
                beta=0.01):
  """Construct the graph to compute ||f(x) - f(x+d)|| num_steps times.

  During each iteration, delta is updated using gradient descent to minimize
  the difference between f(x) and f(x+d).
  """

  dog_mask_resized = img_as_bool(
      resize(dog_mask.astype(np.bool), (loc[2] - loc[0], loc[3] - loc[1])))
  dog_mask_resized = np.expand_dims(dog_mask_resized, axis=2)
  dog_mask_resized = np.array([np.tile(dog_mask_resized, [1, 1, 3])])

  tf.saved_model.loader.load(sess, ['serve'],
                             os.path.join(os.getcwd(), 'models', model_name))
  graph = tf.get_default_graph()
  gdef_1 = tf.graph_util.convert_variables_to_constants(
      sess,
      tf.get_default_graph().as_graph_def(), ['softmax_tensor'],
      variable_names_blacklist=[input_name, func_name])

  def update_delta(delta, i):
    tf.import_graph_def(
        gdef_1,
        input_map={
            input_name:
                tf.get_default_graph().get_tensor_by_name(input_name) + delta
        })
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name(input_name)
    fx = graph.get_tensor_by_name(func_name)
    fxd = graph.get_tensor_by_name('while/import/' + func_name)

    # First loss term: |f(x+d) - f(x)|^2
    loss = tf.norm(fxd - fx)

    # Regularization term to encourage delta with big norm
    loss = loss - alpha * tf.norm(tf.reshape(delta, [-1]))

    # Regularization term to ensure in-range pixel value
    relu = tf.reduce_sum(
        tf.nn.relu(image + delta - image_bounds[0]) +
        tf.nn.relu(tf.fill(tf.shape(image), image_bounds[1]) - image - delta))
    loss = loss + beta * relu
    new_delta = delta - step_size * tf.squeeze(tf.gradients(loss, delta), 0)
    new_delta = tf.pad(
        new_delta[:, loc[0]:loc[2], loc[1]:loc[3], :] * dog_mask_resized,
        tf.constant([[0, 0], [loc[0], 224 - loc[2]], [loc[1], 224 - loc[3]],
                     [0, 0]]))
    return new_delta, i + 1

  def cond(unused_delta, i):
    return tf.less(i, num_steps)

  i = tf.constant(0)
  delta = tf.constant(dog_imgs)
  new_delta, _ = tf.while_loop(cond, update_delta, loop_vars=[delta, i])
  image = tf.get_default_graph().get_tensor_by_name(input_name)
  new_image = image + new_delta
  return tf.stop_gradient(new_image)


def optimize(
    imgs,
    dog_imgs,
    loc,
    mask,
    img_name,
    num_steps=300,
    step_size=500,
    func_name='resnet_model/final_dense:0',
    input_name='input_tensor:0',
):
  with tf.Session(graph=tf.Graph()) as sess:
    graph = tf.get_default_graph()
    new_image_node = build_graph(sess, num_steps, step_size, dog_imgs, loc,
                                 mask)
    input_tensor = graph.get_tensor_by_name(input_name)
    new_image = sess.run(new_image_node, feed_dict={input_tensor: imgs})[0]

    logits = tf.nn.softmax(graph.get_tensor_by_name(func_name))

    img_logit = list(sess.run([logits], feed_dict={input_name: imgs})[0][0])
    new_img_logit = list(
        sess.run([logits], feed_dict={input_name: [new_image]})[0][0])

    fx_node = graph.get_tensor_by_name(func_name)
    fx = sess.run([fx_node], feed_dict={input_name: imgs})[0][0]
    fxd = sess.run([fx_node], feed_dict={input_name: [new_image]})[0][0]
    l2 = np.linalg.norm(fxd - fx)
    print('Logit layer L2 distance = {}'.format(l2))

    fname = img_name[:-4] + '_' + str(int(l2))
    save_img(imgs[0], new_image, fname)


def compute_and_save_delta_patch():
  image_dir = os.path.join(os.getcwd(), 'data', 'bamboo_forest_patch')
  if tf.io.gfile.exists(image_dir):
    tf.io.gfile.rmtree(image_dir)
  tf.io.gfile.makedirs(image_dir)
  imgs = load_imgs(
      sorted(
          tf.io.gfile.glob(
              os.path.join(os.getcwd(), 'data', 'scene_only', 'val',
                           'dog-bamboo_forest*'))))
  img_names = sorted([
      name for name in tf.io.gfile.listdir(
          os.path.join(os.getcwd(), 'data', 'scene_only', 'val'))
      if 'dog-bamboo' in name
  ])
  dog_imgs = load_imgs(
      sorted(
          tf.io.gfile.glob(
              os.path.join(os.getcwd(), 'data', 'scene', 'val',
                           'dog-bamboo_forest*'))))

  lines = tf.gfile.Open(
      os.path.join(os.getcwd(), 'data', 'scene',
                   'val_loc.txt')).readlines()[2000:2100]
  lines = [line.split(' ')[1].split(',') for line in lines]
  locs = np.array(
      [[int(int(l) * float(224) / 128) for l in line] for line in lines])
  masks = np.load(
      os.path.join(os.getcwd(), 'data', 'scene', 'val_mask'),
      allow_pickle=True)[200:300]

  for (img, dog_img, loc, mask, img_name) in zip(imgs, dog_imgs, locs, masks,
                                                 img_names):
    optimize(np.array([img]), np.array([dog_img]), loc, mask, img_name)


if __name__ == '__main__':
  compute_and_save_delta_patch()
