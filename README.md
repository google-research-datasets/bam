# BIM - Benchmark Interpretability Method

This repository contains dataset, models, and metrics for benchmarking
interpretability methods (BIM) described in paper:

*   Title: "BIM: Towards Quantitative Evaluation of Interpretability Methods
    with Ground Truth"
*   Authors: Sherry (Mengjiao) Yang, Been Kim

Upon using this library, please cite:

```
@Article{BIM2019,
  title = {{BIM: Towards Quantitative Evaluation of Interpretability Methods with Ground Truth}},
  author = {Yang, Mengjiao and Kim, Been},
  year = {2019}
}
```

BIM atasets and models will be fully released by the end of June 2019.

## Dataset

The core of BIM dataset, [obj](https://storage.googleapis.com/bim/obj.tar.gz)
and [scene](https://storage.googleapis.com/bim/scene.tar.gz), are constructed by
pasting object pixels from [MSCOCO](http://cocodataset.org) to scene images from
[MiniPlaces](https://github.com/CSAILVision/miniplaces). The obj set and scene
set have object labels and scene labels respectively. In each set,
[val_loc.txt](https://raw.githubusercontent.com/google-research-datasets/bim/master/data/obj/val_loc.txt)
contains row_min, col_min, row_max, col_max of the objects, and
[val_mask](https://raw.githubusercontent.com/google-research-datasets/bim/master/data/obj/val_mask)
contains objects' binary masks.

To compute the BIM metrics, we provide additional image sets described in the
table below.

Download                                                                             | Training | Validation | Usage                                | Description
:----------------------------------------------------------------------------------: | :------: | :--------: | :----------------------------------- | :----------
[obj](https://storage.googleapis.com/bim/obj.tar.gz)                                 | 90,000   | 10,000     | Model contrast                       | Objects and scenes with object labels
[scene](https://storage.googleapis.com/bim/scene.tar.gz)                             | 90,000   | 10,000     | Model contrast<br />Input dependence | Objects and scenes with scene labels
[scene_only](https://storage.googleapis.com/bim/scene_only.tar.gz)                   | 90,000   | 10,000     | Input dependence                     | Images in [scene](https://storage.googleapis.com/bim/scene.tar.gz) with objects removed
[dog_bedroom](https://storage.googleapis.com/bim/dog_bedroom.tar.gz)                 | -        | 200        | Relative model contrast              | Dog in bedroom labeled as bedroom
[bamboo_forest](https://storage.googleapis.com/bim/bamboo_forest.tar.gz)             | -        | 100        | Input independence                   | Scene-only bamboo forest
[bamboo_forest_patch](https://storage.googleapis.com/bim/bamboo_forest_patch.tar.gz) | -        | 100        | Input independence                   | Bamboo forest with functionally insignificant dog patch

## Models

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/dataset_demo.png" width="800">

As shown in the figure above, the
[obj model](https://storage.googleapis.com/bim/models/obj.tar.gz) is trained on
object labels and the
[scene model](https://storage.googleapis.com/bim/models/scene.tar.gz) is trained
on scene labels. We also provide the model trained on scene-only images and a
set of models where the object occurs in a different number of classes. All
models are in TensorFlow's
[SavedModel](https://www.tensorflow.org/guide/saved_model) format.

Download | [obj](https://storage.googleapis.com/bim/models/obj.tar.gz) | [scene](https://storage.googleapis.com/bim/models/scene.tar.gz) | [scene_only](https://storage.googleapis.com/bim/models/scene_only.tar.gz) | [scene1](https://storage.googleapis.com/bim/models/scene1.tar.gz) | [scene2](https://storage.googleapis.com/bim/models/scene2.tar.gz) | [scene3](https://storage.googleapis.com/bim/models/scene3.tar.gz) | [scene4](https://storage.googleapis.com/bim/models/scene4.tar.gz) | [scene5](https://storage.googleapis.com/bim/models/scene5.tar.gz) | [scene6](https://storage.googleapis.com/bim/models/scene6.tar.gz) | [scene7](https://storage.googleapis.com/bim/models/scene7.tar.gz) | [scene8](https://storage.googleapis.com/bim/models/scene8.tar.gz) | [scene9](https://storage.googleapis.com/bim/models/scene9.tar.gz) | [scene10](https://storage.googleapis.com/bim/models/scene10.tar.gz)
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --

## Metrics

BIM metrics compare how interpretability methods perform across models (model
contrast), across inputs to the same model (input dependence), and across
functionally equivalent inputs to the same model (input independence).

### Model contrast scores

Given images that contain both objects and scenes, model contrast measures the
difference in attributions between the model trained on object labels and the
model trained on scene labels.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/mc_demo.png" width="800">

### Input dependence rate

Given a model trained on scene labels, input dependence measures the ratio of
which the objects are attributed as less important compared to when objects are
absent.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/id_demo.png" width="800">

### Input independence rate

Given a model trained on scene-only images, input independence measures the
ratio of which a functionally insignificant patch (e.g., a dog) does not affect
explanations significantly.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/ii_demo.png" width="800">

## Examples

Run `pip install bim` to install python dependencies. You can choose to run
`download.sh` to download the entire dataset and models specified above, or
follow the download link for a particular data or model and extract the tar.gz
to the corresponding `data` or `models` directory. Then you can run

```
python3 metrics.py --metrics=MCS --num_imgs=10
```

to compute the model contrast scores (MCS) over randomly sampled 10 images.
Since computing saliency maps for a large amount of input images can take a
while, we also provide
[precomputed attributions](https://storage.googleapis.com/attr_reprod.tar.gz).
To compute BIM metrics using precomputed attributions, run

```
python3 metrics.py --metrics=MCS --scratch=0
```

## Evaluate TCAV

[TCAV](https://github.com/tensorflow/tcav) is a concept attribution method.
[run_tcav.py](https://raw.githubusercontent.com/google-research-datasets/bim/master/bim/run_tcav.py)
evaluates model contrast of TCAV by comparing the TCAV scores of the dog concept
for the object model versus for the scene model. Run

```
python tcav.py --model=obj
```

to compute the TCAV scores of the default dog concept for the object model.

## Disclaimer

This is not an officially supported Google product.
