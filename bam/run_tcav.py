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

from __future__ import absolute_import

import os
import tensorflow as tf
from absl import app
from absl import flags

from tcav import tcav
from tcav import activation_generator
from tcav import utils
import resnet_model_wrapper

flags.DEFINE_string('model', 'obj',
                    'Name of the model. Should exist in models directory')
FLAGS = flags.FLAGS


def compute_tcav_scores(target='dog_bedroom',
                        random_counterpart='random_counter_part',
                        concepts=['dog_scene']):
  """Compute TCAV scores of a given list of concepts for a ResNet model.

  Computation is done for each block layer and the logit layer.
  """

  base_dir = os.getcwd()
  model_dir = os.path.join(base_dir, 'models', FLAGS.model)
  img_dir = os.path.join(base_dir, 'data/tcav')
  if FLAGS.model == 'obj':
    cat_fpath = os.path.join(base_dir, 'data/tcav', 'obj_categories.txt')
  else:
    cat_fpath = os.path.join(base_dir, 'data/tcav', 'scene_categories.txt')
  working_dir = os.path.join(base_dir, 'tcav_working_dir', FLAGS.model)
  if not tf.gfile.Exists(working_dir):
    tf.gfile.MakeDirs(working_dir)
    tf.gfile.MakeDirs(os.path.join(working_dir, 'activations'))
    tf.gfile.MakeDirs(os.path.join(working_dir, 'cavs'))

  sess = utils.create_session()
  tcav_model_wrapper = resnet_model_wrapper.ResNetModelWrapper(
      sess, model_dir, cat_fpath)
  act_gen = activation_generator.ImageActivationGenerator(
      tcav_model_wrapper,
      img_dir,
      os.path.join(working_dir, 'activations'),
      max_examples=100,
      normalize_image=False)

  bottlenecks = [
      'block_layer1', 'block_layer2', 'block_layer3', 'block_layer4', 'logit'
  ]
  for bottleneck in bottlenecks:
    mytcav = tcav.TCAV(
        sess,
        target,
        concepts, [bottleneck],
        act_gen, [0.1],
        random_counterpart,
        cav_dir=os.path.join(working_dir, 'cavs'),
        num_random_exp=2)
    results = mytcav.run()
    utils.print_results(results, random_counterpart='random_counter_part')


def main(argv):  # pylint: disable=unused-argument
  compute_tcav_scores()


if __name__ == '__main__':
  app.run(main)
