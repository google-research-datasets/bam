#!/bin/bash

for name in obj scene scene_only
do
    tar -xvzf $PWD/models/$name.tar.gz -C $PWD/models/
done

for i in {1..10}
do
    tar -xvzf $PWD/models/scene$i.tar.gz -C $PWD/models/
done
