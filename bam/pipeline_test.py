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

from tensorflow.python.platform import googletest
from metrics import load_pos_neg_attr
from metrics import compute_and_save_attr
import random
import os

class PipelineTest(googletest.TestCase):

  def test_compute_and_load_attr_correct(self):
      models = ['obj', 'scene']
      datas = ['scene', 'obj']
      num_imgs = 3

      # Sample num_imgs images to compute attributions
      random.seed(0)
      indices = random.sample(range(100), num_imgs)
      
      compute_and_save_attr(models[0], datas[0], indices, 10)
      compute_and_save_attr(models[1], datas[1], indices, 10)

      sal_dir_pos = os.path.join(os.getcwd(), 'sal_scratch', models[0] + '-' + datas[0])
      sal_dir_neg = os.path.join(os.getcwd(), 'sal_scratch', models[1] + '-' + datas[1])

      # Saliency maps directory contains the right files
      self.assertEqual(len(os.listdir(sal_dir_pos)), num_imgs)
      self.assertEqual(len(os.listdir(sal_dir_neg)), num_imgs)
      self.assertEqual(sorted(os.listdir(sal_dir_pos)), sorted(os.listdir(sal_dir_neg)))

      attr_pos, attr_neg = load_pos_neg_attr(models[0], datas[0],
                                             models[1], datas[1])

      attr_dir_pos = os.path.join(os.getcwd(), 'attr_scratch', models[0] + '-' + datas[0])
      attr_dir_neg = os.path.join(os.getcwd(), 'attr_scratch', models[1] + '-' + datas[1])

      # Attributions directory contains the right files      
      self.assertEqual(len(os.listdir(attr_dir_pos)), num_imgs)
      self.assertEqual(len(os.listdir(attr_dir_neg)), num_imgs)
      self.assertEqual(sorted(os.listdir(attr_dir_pos)), sorted(os.listdir(attr_dir_neg)))

      # Attribution is higher in obj model than in scene model for these examples
      self.assertEqual(attr_pos.shape, attr_neg.shape)
      method_indices = [6, 0, 1, 4, 5, 10, 2, 8]
      for i in range(num_imgs):
          for j in method_indices:
              self.assertLess(attr_neg[i,j], attr_pos[i, j])    

if __name__ == '__main__':
    googletest.main()
