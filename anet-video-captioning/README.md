# Quick Start (Video Captioning)

The following guideline shows how to train and evaluate on [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities).

## Training

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

### Evaluation with provided baseline

#### Download the checkpoints provided

- [baseline](https://www.dropbox.com/s/lqk6oyx8tnktoqt/baseline.zip?dl=1) (242MB)
- [cyclical](https://www.dropbox.com/s/y4d46wot95gxeql/cyclical.zip?dl=1) (242MB)

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

You should get the following performance:

```python
Bleu_1: 23.139
Bleu_4: 2.377
METEOR: 10.870
CIDEr: 47.399
SPICE: 15.226
F1_all: 0.0401
F1_all_per_sent: 0.1017
F1_loc: 0.1301
F1_loc_per_sent: 0.3413
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

You should get the following performance:

```python
Bleu_1: 23.643
Bleu_4: 2.501
METEOR: 11.148
CIDEr: 46.872
SPICE: 15.243
F1_all: 0.0477
F1_all_per_sent: 0.1250
F1_loc: 0.1599
F1_loc_per_sent: 0.4241
```

## Test server submission

Change the `val_split` and `densecap_references` accordingly as follows. 
Check the instruction provided in [GVD GitHub repo](https://github.com/facebookresearch/grounded-video-description#inference-and-testing) for more infomation regarding test (or hidden test) server submission.

```yml
# use validation set as default
val_split: validation
densecap_references: ["anet_entities_val_1.json", "anet_entities_val_2.json"]

# change to the following two lines for test set submission instead
val_split: testing
densecap_references: ["anet_entities_test_1.json", "anet_entities_test_2.json"]
```

You might want to change `*testing*.json` to `*hidden_test*.json` for submission of hidden test set.
