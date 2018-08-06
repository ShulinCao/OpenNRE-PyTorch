# OpenNRE-PyTorch

An open-source framework for neural relation extraction implemented in PyTorch.

Contributed by [Shulin Cao](https://github.com/ShulinCao), [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)

## Overview

It is a PyTorch-based framwork for easily building relation extraction models. We divide the pipeline of relation extraction into four parts, which are embedding, encoder, selector and classifier. For each part we have implemented several methods.

* Embedding
  * Word embedding
  * Position embedding
  * Concatenation method
* Encoder
  * PCNN
  * CNN
* Selector
  * Attention
  * Maximum
  * Average
* Classifier
  * Softmax loss function
  * Output
  
All those methods could be combined freely. 

We also provide fast training and testing codes. You could change hyper-parameters or appoint model architectures by using Python arguments. A plotting method is also in the package.

This project is under MIT license.

## Requirements

- Python (>=2.7)
- PyTorch (>=0.3.1)
	- CUDA (>=8.0) if you are using gpu
- Matplotlib (>=2.0.0)
- scikit-learn (>=0.18)

## Installation

1. Install PyTorch
2. Clone the OpenNRE repository:
  ```bash
  git clone https://github.com/ShulinCao/OpenNRE-PyTorch
  ```
3. Download NYT dataset from `https://drive.google.com/file/d/1BnyXMJ71jM0kxyJUlXa5MHf-vNjE57i-/view?usp=sharing`
4. Extract dataset to `./origin_data`
```
tar xvf origin_data.tar
```

## Results

### PCNN-based methods

![pcnn precision-recall curve](https://github.com/ShulinCao/OpenNRE-PyTorch/blob/master/results/pcnn.png)

method | auc | F1 score
---- | ---- | ----
pcnn_att | **0.412650381932** | **0.450063211125**
pcnn_max | 0.40546481516 | 0.441361916772
pcnn_ave | 0.403195681149 | 0.440381148596

### Compare PCNN & CNN

![pcnn & cnn precision-recall curve](https://github.com/ShulinCao/OpenNRE-PyTorch/blob/master/results/pcnn_cnn.png)

## Quick Start

### Process Data

```bash
python gen_data.py
```
The processed data will be stored in `./data`

### Train Model
```
python train.py --model_name pcnn_att
```

The arg `model_name` appoints model architecture, and `pcnn_att` is the name of one of our models. All available models are in `./model`. About other arguments please refer to `./train.py`. Once you start training, all checkpoints are stored in `./checkpoint`.

### Test Model
```bash
python test.py --model_name pcnn_att
```

Same usage as training. When finishing testing, the best checkpoint's corresponding pr-curve data will be stored in `./test_result`.

### Plot
```bash
python draw_plot.py PCNN_ATT
```

The plot will be saved as `./test_result/pr_curve.png`. You could appoint several models in the arguments, like `python draw_plot.py PCNN_ATT PCNN_MAX PCNN_AVE`, as long as there are these models' results in `./test_result`.

## Build Your Own Model

Not only could you train and test existing models in our package, you could also build your own model or add methods to the four basic modules. When adding a new model, you could create a python file in `./models` having the same name as the model and implement it like following:

```python
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *
from .Model import Model
class PCNN_ATT(Model):
  def __init__(self, config):
    super(PCNN_ATT, self).__init__(config)
    self.encoder = PCNN(config)
    self.selector = Attention(config, config.hidden_size * 3)
```

Then you can train, test and plot!

