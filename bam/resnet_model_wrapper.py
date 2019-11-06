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

from tcav import model
import tensorflow as tf


class ResNetModelWrapper(model.ImageModelWrapper):

  def __init__(self, sess, model_dir, categories_file):
    """
        Args:
          categories_file: file path that contains alphabetically sorted
            categories and their corresponding index. e.g. /a/abbey 0
    """
    self.model_name = "ResNet"
    self.sess = sess
    image_shape = (224, 224, 3)
    super(ResNetModelWrapper, self).__init__(image_shape)

    tf.saved_model.loader.load(sess, ["serve"], model_dir)
    self.labels = [
        line.split(" ")[0]
        for line in tf.gfile.Open(categories_file).read().splitlines()
    ]

    node_dict = {
        "input": "input_tensor",
        "block_layer1": "resnet_model/block_layer1",
        "block_layer2": "resnet_model/block_layer2",
        "block_layer3": "resnet_model/block_layer3",
        "block_layer4": "resnet_model/block_layer4",
        "logit": "resnet_model/final_dense"
    }
    self.bottlenecks_tensors = {}
    for endpoint in node_dict:
      self.bottlenecks_tensors[endpoint] = sess.graph.get_operation_by_name(
          node_dict[endpoint]).outputs[0]
    self.bottleneck_names = self.bottlenecks_tensors.keys()
    self.ends = {}
    self.ends["input"] = self.bottlenecks_tensors["input"]
    self.ends["logit"] = self.bottlenecks_tensors["logit"]
    self.ends["prediction"] = self.ends["logit"]

    with sess.graph.as_default():
      self.y_input = tf.placeholder(tf.int64, shape=[None])
      self.pred = tf.expand_dims(self.ends["prediction"][0], 0)
      self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=tf.one_hot(self.y_input, len(self.labels)),
              logits=self.pred))
    self._make_gradient_tensors()

  def label_to_id(self, label):
    return self.labels.index(label)

  def id_to_label(self, idx):
    return self.labels[idx]
