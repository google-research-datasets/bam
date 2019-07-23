#!/bin/bash

mkdir $PWD/data
mkdir $PWD/data/coco
mkdir $PWD/data/miniplaces

curl -sS 'http://images.cocodataset.org/zips/train2017.zip' > images.zip && \
unzip images.zip -d $PWD/data/coco/images/ && \
rm images.zip

curl -sS 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip' > ann.zip && \
unzip ann.zip -d $PWD/data/coco/ && \
rm ann.zip

wget -qO- 'http://miniplaces.csail.mit.edu/data/data.tar.gz' | tar -xz -C $PWD/data/miniplaces
rm -rf $PWD/data/miniplaces/objects
