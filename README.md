# BIM - Benchmark Interpretability Method

This repository contains dataset, models, and metrics for benchmarking
interpretability methods (BIM) described in paper [BIM: Towards Quantitative Evaluation of Interpretability Methods
with Ground Truth](https://arxiv.org/abs/1907.09701). Upon using this library, please cite:

```
@Article{BIM2019,
  title = {{BIM: Towards Quantitative Evaluation of Interpretability Methods with Ground Truth}},
  author = {Yang, Mengjiao and Kim, Been},
  journal   = {CoRR},
  volume    = {abs/1907.09701},
  year = {2019}
}
```

## Setup

Run the following from the home directory of this repository to install python
dependencies, download BIM models, download [MSCOCO](http://cocodataset.org) and
[MiniPlaces](https://github.com/CSAILVision/miniplaces), and construct BIM
dataset.

```
pip install bim
source scripts/download_models.sh
source scripts/download_datasets.sh
python scripts/construct_bim_dataset.py
```

## Dataset

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/dataset_demo.png" width="800">

Images in `data/obj` and `data/scene` are the same but have object and scene
labels respectively, as shown in the figure above. `val_loc.txt` records the
top-left and bottom-right corner of the object and `val_mask` has the binary
masks of the object in the validation set. Additional sets and their usage are
described in the table below.

Name                  | Training | Validation | Usage                             | Description
:---------------------------------------------------------------------------------| :------: | :--------: | :----------------------------------- | :----------
`obj`                 | 90,000   | 10,000     | Model contrast                    | Objects and scenes with object labels
`scene`               | 90,000   | 10,000     | Model contrast & Input dependence | Objects and scenes with scene labels
`scene_only`          | 90,000   | 10,000     | Input dependence                  | Scene-only images with scene labels
`dog_bedroom`         | -        | 200        | Relative model contrast           | Dog in bedroom labeled as bedroom
`bamboo_forest`       | -        | 100        | Input independence                | Scene-only images of bamboo forest
`bamboo_forest_patch` | -        | 100        | Input independence                | Bamboo forest with functionally insignificant dog patch

## Models

Models in `models/obj`, `models/scene`, and `models/scene_only` are trained on
`data/obj`, `data/scene`, and `data/scene_only` respectively. Models in
`models/scenei` for `i` in `{1...10}` are trained on images where dog is added
to `i` scene classes, and the rest scene classes do not contain any added
objects. All models are in TensorFlow's
[SavedModel](https://www.tensorflow.org/guide/saved_model) format.

## Metrics

BIM metrics compare how interpretability methods perform across models (model
contrast), across inputs to the same model (input dependence), and across
functionally equivalent inputs (input independence).

### Model contrast scores

Given images that contain both objects and scenes, model contrast measures the
difference in attributions between the model trained on object labels and the
model trained on scene labels.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/mc_demo.png" width="800">

### Input dependence rate

Given a model trained on scene labels, input dependence measures the percentage
of inputs where the addition of objects results in the region being attributed
as less important.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/id_demo.png" width="800">

### Input independence rate

Given a model trained on scene-only images, input independence measures the
percentage of inputs where a functionally insignificant patch (e.g., a dog) does
not affect explanations significantly.

<img src="https://raw.githubusercontent.com/google-research-datasets/bim/master/figures/ii_demo.png" width="800">

## Evaluate saliency methods

To compute model contrast score (MCS) over randomly selected 10 images, you can
run

```
python bim/metrics.py --metrics=MCS --num_imgs=10
```

To compute input dependence rate (IDR), change `--metrics` to `IDR`. To compute
input independence rate (IIR), you need to first constructs a set of
functionally insignificant patches by running

```
python scripts/construct_delta_patch.py
```

and then evaluate IIR by running

```
python bim/metrics.py --metrics=IIR --num_imgs=10
```

## Evaluate TCAV

[TCAV](https://github.com/tensorflow/tcav) is a global concept attribution
method whose MCS can be measured by comparing the TCAV scores of a particular
object concept for the object model and the scene model. Run the following to
compute the TCAV scores of the dog concept for the object model.

```
python bim/run_tcav.py --model=obj
```

## Disclaimer

This is not an officially supported Google product.
