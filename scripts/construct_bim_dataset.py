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
import numpy as np
from PIL import Image
import tensorflow as tf
from pycocotools.coco import COCO
import skimage.io as io
import shutil
from multiprocessing import dummy as multiprocessing
import re
import random

NUM_IMAGES_PER_CLASS = 1000
TRAIN_VAL_RATIO = 0.9
OBJ_NAMES = [
    'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
    'toilet', 'truck', 'zebra'
]
SCENE_NAMES = [
    'bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit',
    'corn_field', 'laundromat', 'runway', 'ski_slope', 'track/outdoor'
]


def extract_coco_objects():
  coco_dir = os.path.join(os.getcwd(), 'data', 'coco')
  data_type = 'train2017'
  ann_file = '{}/annotations/instances_{}.json'.format(coco_dir, data_type)
  output_dir = os.path.join(coco_dir, 'segments')

  coco = COCO(ann_file)
  for obj_name in OBJ_NAMES:
    masks = []
    dst_dir = os.path.join(output_dir, obj_name)
    if not tf.io.gfile.exists(dst_dir):
      tf.io.gfile.makedirs(dst_dir)
    cat_ids = coco.getCatIds(catNms=[obj_name])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(img_ids)
    num_imgs = 0
    for img in imgs:
      try:
        I = io.imread(
            tf.io.gfile.GFile(
                '%s/images/%s/%s' % (coco_dir, data_type, img['file_name']),
                'rb'))
        org_h, org_w = I.shape[0], I.shape[1]
        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        max_area = 0
        max_ann = None
        for ann in anns:
          if ann['area'] > max_area:
            max_ann = ann
            max_area = ann['area']
        ann = max_ann
        base_x, base_y, bbox_w, bbox_h = ann['bbox']
        mask2d = coco.annToMask(ann)
        mask = np.stack((mask2d,) * 3, axis=-1)
        cropped_imgarr = (I * mask)[int(base_y):int(base_y + bbox_h),
                                    int(base_x):int(base_x + bbox_w), :]
        cropped_img = Image.fromarray(cropped_imgarr).convert('RGBA')
        newData = []
        for item in cropped_img.getdata():
          if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
          else:
            newData.append(item)
        cropped_img.putdata(newData)
        if img['file_name'][-3:] == 'jpg':
          fname = img['file_name'][:-3] + 'png'
        cropped_img.save(
            tf.io.gfile.GFile(os.path.join(dst_dir, fname[:-4] + '.png'), 'w'),
            format='png')
        masks.append(
            (img['file_name'], mask2d[int(base_y):int(base_y + bbox_h),
                                      int(base_x):int(base_x + bbox_w)]))

        num_imgs += 1
        if num_imgs == NUM_IMAGES_PER_CLASS:
          break
      except:
        continue

    # Save the last 100 object masks used for validation
    masks.sort()
    save_start = int(NUM_IMAGES_PER_CLASS * TRAIN_VAL_RATIO)
    print(len(masks))
    print(len(masks[save_start:]))
    #print(masks[save_start:][0][1])
    np.save(
        tf.io.gfile.GFile(os.path.join(dst_dir, 'val_mask'), 'w'),
        np.array([m[1] for m in masks[save_start:]]))


def rename_miniplaces_scenes():
  miniplaces_dir = os.path.join(os.getcwd(), 'data', 'miniplaces')
  output_dir = os.path.join(miniplaces_dir, 'scene_subset')
  for scene_name in SCENE_NAMES:
    src_dir = os.path.join(miniplaces_dir, 'images', 'train', scene_name[0],
                           scene_name)
    dst_dir = os.path.join(output_dir, scene_name.split('/')[0])
    shutil.copytree(src_dir, dst_dir)


def paste_objects_to_scenes():
  obj_dir = os.path.join(os.getcwd(), 'data', 'coco', 'segments')
  sce_dir = os.path.join(os.getcwd(), 'data', 'miniplaces', 'scene_subset')

  obj_raw = os.path.join(os.getcwd(), 'data', 'obj_all')
  sce_raw = os.path.join(os.getcwd(), 'data', 'scene_all')
  sce_only = os.path.join(os.getcwd(), 'data', 'scene_only_all')
  objs = tf.io.gfile.listdir(obj_dir)
  sces = tf.io.gfile.listdir(sce_dir)
  for o in objs:
    tf.io.gfile.makedirs(os.path.join(obj_raw, o))
  for s in sces:
    tf.io.gfile.makedirs(os.path.join(sce_raw, s))
    tf.io.gfile.makedirs(os.path.join(sce_only, s))

  objs = tf.io.gfile.listdir(obj_dir)
  sces = tf.io.gfile.listdir(sce_dir)
  locs = []
  for obj in objs:
    obj_fnames = sorted(tf.io.gfile.listdir(os.path.join(
        obj_dir, obj)))[:NUM_IMAGES_PER_CLASS]
    for sce in sces:
      sce_fnames = sorted(tf.io.gfile.listdir(os.path.join(sce_dir, sce)))
      for i in range(NUM_IMAGES_PER_CLASS):
        obj_fname, sce_fname = obj_fnames[i], sce_fnames[i]
        sce_img = Image.open(
            tf.io.gfile.GFile(os.path.join(sce_dir, sce, sce_fname),
                              'rb')).convert('RGBA')
        scene_w, scene_h = sce_img.size
        obj_img = Image.open(
            tf.io.gfile.GFile(os.path.join(obj_dir, obj, obj_fname),
                              'rb')).convert('RGBA')
        obj_w, obj_h = obj_img.size

        # Resize the obj to fit into the scene. The resized object is
        # between 1/3 and 1/2 of the scene
        resize_lo, resize_hi = np.sqrt(3), np.sqrt(2)
        if float(obj_w) / scene_w > float(obj_h) / scene_h:
          new_obj_w = np.random.randint(
              int(scene_w / resize_lo), int(scene_w / resize_hi))
          new_obj_h = int(float(new_obj_w) / obj_w * obj_h)
        else:
          new_obj_h = np.random.randint(
              int(scene_h / resize_lo), int(scene_h / resize_hi))
          new_obj_w = int(float(new_obj_h) / obj_h * obj_w)

        # Randomly generate a location to place the obj
        row = np.random.randint(0, scene_h - new_obj_h)
        col = np.random.randint(0, scene_w - new_obj_w)

        obj_img = obj_img.resize((new_obj_w, new_obj_h), Image.BILINEAR)

        fname = obj + '-' + sce + '-' + str(i).zfill(4) + '.jpg'
        # Save scene to scene_only folder
        sce_img.convert('RGB').save(
            tf.io.gfile.GFile(os.path.join(sce_only, sce, fname), 'w'),
            format='jpeg')

        # Save obj-scene to both obj and scene folders
        sce_img.paste(obj_img, (col, row), obj_img)
        sce_img = sce_img.convert('RGB')
        sce_img.save(
            tf.io.gfile.GFile(os.path.join(sce_raw, sce, fname), 'w'),
            format='jpeg')
        sce_img.save(
            tf.io.gfile.GFile(os.path.join(obj_raw, obj, fname), 'w'),
            format='jpeg')

        locs.append(
            fname + ' ' +
            ','.join(map(str, [row, col, row + new_obj_h, col + new_obj_w])))
  with tf.io.gfile.GFile(os.path.join(sce_only, 'loc.txt'), 'w') as f:
    f.write('\n'.join(sorted(locs)))
  with tf.io.gfile.GFile(os.path.join(sce_raw, 'loc.txt'), 'w') as f:
    f.write('\n'.join(sorted(locs)))
  with tf.io.gfile.GFile(os.path.join(obj_raw, 'loc.txt'), 'w') as f:
    f.write('\n'.join(sorted(locs)))


def divide_train_val():
  all_masks = []
  for dir_name in ['obj', 'scene', 'scene_only']:
    image_dir = os.path.join(os.getcwd(), 'data', dir_name)
    if tf.io.gfile.exists(image_dir):
      tf.io.gfile.rmtree(image_dir)
    tf.io.gfile.makedirs(image_dir)
    train_dir = os.path.join(image_dir, 'train')
    tf.io.gfile.makedirs(train_dir)
    val_dir = os.path.join(image_dir, 'val')
    tf.io.gfile.makedirs(val_dir)

    loc_table = {}
    lines = tf.io.gfile.GFile(os.path.join(image_dir + '_all',
                                           'loc.txt')).readlines()
    for line in lines:
      loc_table[line.split(' ')[0]] = line.split(' ')[1]

    classes = tf.io.gfile.listdir(image_dir + '_all')
    classes.remove('loc.txt')
    classes.sort()

    pool = multiprocessing.Pool(100)
    val_lines = []
    loc_lines = []
    for i in range(len(classes)):
      a_class = classes[i]
      if dir_name == 'obj':
        all_masks.append(
            np.load(
                os.path.join(os.getcwd(), 'data', 'coco', 'segments', a_class,
                             'val_mask'),
                allow_pickle=True))

      files = np.array(
          tf.io.gfile.listdir(os.path.join(image_dir + '_all', a_class)))
      train_files = []
      val_files = []
      for f in files:
        if re.search('9[0-9]{2}\.jpg', f):
          val_files.append(f)
        else:
          train_files.append(f)
      val_files.sort()
      tf.io.gfile.makedirs(os.path.join(train_dir, a_class))
      pool.map(
          lambda f: tf.io.gfile.copy(
              os.path.join(image_dir + '_all', a_class, f),
              os.path.join(train_dir, a_class, f)), train_files)
      pool.map(
          lambda f: tf.io.gfile.copy(
              os.path.join(image_dir + '_all', a_class, f),
              os.path.join(val_dir, f)), val_files)
      val_lines += [f + ' ' + str(i) + '\n' for f in val_files]
      loc_lines += [f + ' ' + loc_table[f] for f in val_files]

    val_txt_path = os.path.join(image_dir, 'val.txt')
    with tf.io.gfile.GFile(val_txt_path, 'w') as f:
      f.write(''.join(sorted(val_lines)))

    with tf.io.gfile.GFile(os.path.join(image_dir, 'val_loc.txt'), 'w') as f:
      f.write(''.join(sorted(loc_lines)))

    # Copy over the object masks for the validation set
    np.save(
        tf.io.gfile.GFile(os.path.join(image_dir, 'val_mask'), 'w'),
        np.concatenate(all_masks))
    shutil.copy(
        os.path.join(image_dir + '_all', 'loc.txt'),
        os.path.join(image_dir, 'loc.txt'))
    tf.io.gfile.rmtree(image_dir + '_all')


def save_dog_bedroom_or_bamboo_forest(src_dir_name, dst_dir_name, fname_prefix,
                                      class_label):
  image_dir = os.path.join(os.getcwd(), 'data', dst_dir_name)
  if tf.io.gfile.exists(image_dir):
    tf.io.gfile.rmtree(image_dir)
  tf.io.gfile.makedirs(image_dir)
  tf.io.gfile.makedirs(os.path.join(image_dir, 'val'))
  src_fpaths = tf.io.gfile.glob(
      os.path.join(os.getcwd(), 'data', src_dir_name, 'val',
                   fname_prefix + '*'))
  fnames = [
      os.path.relpath(f, os.path.join(os.getcwd(), 'data', src_dir_name, 'val'))
      for f in src_fpaths
  ]
  dst_fpaths = [os.path.join(image_dir, 'val', f) for f in fnames]
  for src, dst in zip(src_fpaths, dst_fpaths):
    tf.io.gfile.copy(src, dst)
  with open(os.path.join(image_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join([fn + ' ' + str(class_label) for fn in sorted(fnames)]))
  with open(os.path.join(os.getcwd(), 'data', src_dir_name,
                         'val_loc.txt')) as f:
    lines = f.readlines()
    lines = [line for line in lines if fname_prefix in line]
  with open(os.path.join(image_dir, 'val_loc.txt'), 'w') as f:
    f.write(''.join(lines))
  tf.io.gfile.copy(
      os.path.join(os.getcwd(), 'data', 'coco', 'segments', 'dog', 'val_mask'),
      os.path.join(image_dir, 'val_mask'))


def save_tcav_images():
  tcav_data_dir = os.path.join(os.getcwd(), 'data', 'tcav')
  if tf.io.gfile.exists(tcav_data_dir):
    tf.io.gfile.rmtree(tcav_data_dir)
  tf.io.gfile.makedirs(tcav_data_dir)
  shutil.copytree(
      os.path.join(os.getcwd(), 'data', 'dog_bedroom', 'val'),
      os.path.join(tcav_data_dir, 'dog_bedroom'))

  tf.io.gfile.makedirs(os.path.join(tcav_data_dir, 'dog_scene'))
  src_fpaths = tf.io.gfile.glob(
      os.path.join(os.getcwd(), 'data', 'scene', 'val', 'dog*'))
  fnames = [
      os.path.relpath(f, os.path.join(os.getcwd(), 'data', 'scene', 'val'))
      for f in src_fpaths
  ]
  fnames = [fnames[i] for i in random.sample(range(len(fnames)), 100)]
  src_fpaths = [
      os.path.join(os.getcwd(), 'data', 'scene', 'val', f) for f in fnames
  ]
  dst_fpaths = [os.path.join(tcav_data_dir, 'dog_scene', f) for f in fnames]
  for src, dst in zip(src_fpaths, dst_fpaths):
    tf.io.gfile.copy(src, dst)

  tf.io.gfile.makedirs(os.path.join(tcav_data_dir, 'random500_0'))
  tf.io.gfile.makedirs(os.path.join(tcav_data_dir, 'random500_1'))
  src_fpaths = tf.io.gfile.glob(
      os.path.join(os.getcwd(), 'data', 'scene_only', 'val', '*.jpg'))
  fnames = [
      os.path.relpath(f, os.path.join(os.getcwd(), 'data', 'scene_only', 'val'))
      for f in src_fpaths
  ]
  fnames = [fnames[i] for i in random.sample(range(len(fnames)), 200)]
  for i in range(len(fnames)):
    if i < len(fnames) / 2:
      tf.io.gfile.copy(
          os.path.join(os.getcwd(), 'data', 'scene_only', 'val', fnames[i]),
          os.path.join(tcav_data_dir, 'random500_0', fnames[i]))
    else:
      tf.io.gfile.copy(
          os.path.join(os.getcwd(), 'data', 'scene_only', 'val', fnames[i]),
          os.path.join(tcav_data_dir, 'random500_1', fnames[i]))

  tf.io.gfile.makedirs(os.path.join(tcav_data_dir, 'random_counter_part'))
  fnames = tf.io.gfile.listdir(
      os.path.join(os.getcwd(), 'data', 'coco', 'images', 'val'))
  fnames = [fnames[i] for i in random.sample(range(len(fnames)), 100)]
  for fname in fnames:
    tf.io.gfile.copy(
        os.path.join(os.getcwd(), 'data', 'coco', 'images', 'val', fname),
        os.path.join(tcav_data_dir, 'random_counter_part', fname))

  with open(
      os.path.join(os.getcwd(), 'data', 'tcav', 'obj_categories.txt'),
      'w') as f:
    lines = []
    for i in range(len(OBJ_NAMES)):
      if 'dog' in OBJ_NAMES[i]:
        lines.append('dog_bedroom' + ' ' + str(i))
      else:
        lines.append(OBJ_NAMES[i] + ' ' + str(i))
    f.write('\n'.join(lines))
  with open(
      os.path.join(os.getcwd(), 'data', 'tcav', 'scene_categories.txt'),
      'w') as f:
    lines = []
    for i in range(len(SCENE_NAMES)):
      if 'bedroom' in SCENE_NAMES[i]:
        lines.append('dog_bedroom' + ' ' + str(i))
      else:
        lines.append(SCENE_NAMES[i].split('/')[0] + ' ' + str(i))
    f.write('\n'.join(lines))


if __name__ == '__main__':
  extract_coco_objects()
  rename_miniplaces_scenes()
  paste_objects_to_scenes()
  divide_train_val()
  save_dog_bedroom_or_bamboo_forest('scene', 'dog_bedroom', 'dog-bedroom', 1)
  save_dog_bedroom_or_bamboo_forest('scene_only', 'bamboo_forest',
                                    'dog-bamboo_forest', 0)
  save_tcav_images()
