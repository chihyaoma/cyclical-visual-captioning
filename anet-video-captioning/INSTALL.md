# Cyclical Video Captioning

## Quick Start

Clone the repo recursively:

```shell
git clone --recursive git@github.com:chihyaoma/cyclical-visual-captioning.git
```

### Installation

<!-- ```shell
MINICONDA_ROOT=[to your Miniconda root directory]
conda env create -f cfgs/conda_env_cyclical.yml --prefix $MINICONDA_ROOT/envs/cyclical_pytorch1.4
conda activate cyclical_pytorch1.4
``` -->

```shell
# create conda env
conda create -n cyclical python=3.7

conda create -n cyclical-torch1.3 python=3.6
conda create -n cyclical-torch1.1 python=3.6


# activate the enviorment
conda activate cyclical
conda activate cyclical-torch1.3
conda activate cyclical-torch1.1


# install addtional packages
pip install -r requirements.txt

# install pytorch

# (testing if installing from conda -c pytorch will fix segmentation fault issue. Still has segmentation fault. torchtext from pip still doesn't work)
conda install pytorch torchvision -c pytorch  # this will have torch 1.4.0, it's different from installing from pip 
conda install -c pytorch torchtext


(segmentation fault) conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=10.0 -c pytorch  
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch  # for MacOS

# install torchtext
pip install torchtext

### Download NLTK POS tagger
python
>>> import nltk
>>> nltk.download('averaged_perceptron_tagger')
exit()
```

### Download everything

Simply run the following command to download all the data and pre-trained models (total 216GB):

```shell
bash tools/download_all.sh
```

### Local development on MacOS

If you are just like me, who likes to enjoy the fast speed that local development brings you, this code is prepared to make local code development much easier for you. It allows you to develop your next research idea on a laptop (e.g., MacBook) with limited computational resources and storage (i.e., the storage space is too small on a laptop).

Please see the following instructions ...

### Known issues

#### PyTorch older than 1.3

Older PyTorch version (<1.3) does not support the usage of converting Tensor to bool type by `.bool()`.

```shell
AttributeError: 'Tensor' object has no attribute 'bool'
```


#### Pillow on older PyTorch

Pillow released version 7.0.0 early 2020, and this would break on older PyTorch versions (PyTorch < 1.4.0).

```shell
# if you see this error
ImportError: cannot import name 'PILLOW_VERSION'

# install pillow with version older than 7.0.0
pip install "pillow<7"
```
