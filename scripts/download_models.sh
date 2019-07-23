#!/bin/bash

mkdir $PWD/models

declare -a arr=(obj)

base_url=https://storage.googleapis.com/benchmark-intp-methods/models/

for name in obj scene scene_only
do
    curl -sS $base_url$name.zip > $PWD/models/$name.zip && \
    unzip $PWD/models/$name.zip -d $PWD/models/ && \
    rm $PWD/models/$name.zip
done

for i in {1..10}
do
    curl -sS "{$base_url}scene$i".zip > $PWD/models/scene$i.zip && \
    unzip $PWD/models/scene$i.zip -d $PWD/models/ && \
    rm $PWD/models/scene$i.zip
done
