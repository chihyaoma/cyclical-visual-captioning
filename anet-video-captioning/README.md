# Quick Start (Video Captioning)

## Installation

Please follow the installation guide in [INSTALL.md](INSTALL.md) to install PyTorch and other necessary packages.

## Training

The following guideline shows how to train and evaluate on [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities).

### Train baseline model

Use the following command to start the training of the baseline model.

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/baseline.yml'
```

Note that you can of course use `CUDA_VISIBLE_DEVICES=0,1,2,3` to use 4 GPUs, but there is a segmentation fault issue caused by either CUDA or PyTorch where running 4 2080 Ti GPUs.
See [INSTALL.md](INSTALL.md) for more detail and how to fix it.

### Train with Cyclical regimen

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/cyclical.yml'
```

## Evaluation

You can use the `baseline.yml` and `cyclical.yml` to run evaluation.
All you need is to change the following the `True`.
This will resume the best checkpoint found (based on CIDEr score) and run evaluation only.
It will also run caption and grounding evaluation metrics.

```yaml
eval_obj_grounding: True
inference_only: True
resume: True
load_best_score: True
```

### Evaluation with provided checkpoints

#### Download the checkpoints provided

- [baseline](https://www.dropbox.com/s/lqk6oyx8tnktoqt/baseline.zip?dl=1) (242MB)
- [cyclical](https://www.dropbox.com/s/y4d46wot95gxeql/cyclical.zip?dl=1) (242MB)

### Evaluation with baseline

To run evaluation using the baseline we provided:

```yaml
# make sure the exp_name correspondes to the folder name of provided baseline
exp_name: baseline

# change all the following to True as mentioned above
eval_obj_grounding: True
inference_only: True
resume: True
load_best_score: True
```

Start evaluation process by running:

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/baseline.yml'
```

### Evaluation with Cyclical model

To run evaluation using the checkpoint of Cyclical model we provided:

```yaml
# make sure the exp_name correspondes to the folder name of provided baseline
exp_name: cyclical

# change all the following to True as mentioned above
eval_obj_grounding: True
inference_only: True
resume: True
load_best_score: True
```

Start evaluation process by running:

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/cyclical.yml'
```

You should get the following performance for `baseline` and `cyclical` using the checkpoints we provided:

|          | B@1    | B@4   | METEOR | CIDEr  | SPICE  | F1_all | F1_all_per_sent | F1_loc | F1_loc_per_sent |
|----------|--------|-------|--------|--------|--------|--------|-----------------|--------|-----------------|
| Baseline | 23.227 | 2.386 | 10.914 | 47.602 | 15.297 | 0.0400 | 0.1014          | 0.1310 | 0.3405          |
| Cyclical | 23.743 | 2.506 | 11.194 | 47.027 | 15.312 | 0.0478 | 0.1249          | 0.1612 | 0.4237          |

## Test server submission

Change the `val_split`, `densecap_references`, and `grd_reference` in the configure files (under `cfgs/`) accordingly as follows.
Check the instruction provided in [GVD GitHub repo](https://github.com/facebookresearch/grounded-video-description#inference-and-testing) for more infomation regarding test (or hidden test) server submission.

```yml
# use validation set as default
val_split: validation
densecap_references: ["anet_entities_val_1.json", "anet_entities_val_2.json"]
grd_reference: "tools/anet_entities/data/anet_entities_cleaned_class_thresh50_trainval.json"

# change to the following two lines for test set submission instead
val_split: testing
densecap_references: ["anet_entities_test_1.json", "anet_entities_test_2.json"]
grd_reference: "tools/anet_entities/data/anet_entities_cleaned_class_thresh50_test_skeleton.json"

# change to the following two lines for hidden test set submission instead
val_split: hidden_test
grd_reference: "tools/anet_entities/data/anet_entities_cleaned_class_thresh50_hidden_test_skeleton.json"
```
