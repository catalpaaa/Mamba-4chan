# Mamba 4chan

## About

The Kingdom of the Crystal Kek, the sequel to Raiders of the Lost Kek. The legendary GPT-4chan is returned with [selective SSM](https://arxiv.org/abs/2312.00752).

## Installation

We provided a simple [setup.sh](setup.sh) to install the Conda environment. You need to satisfy the following prerequisite:

- Linux
- NVIDIA GPU
- CUDA 12+ supported GPU driver
- Miniforge

Then, simply run `source ./setup.sh` to get started.

## Dataset

We utilized the [Raiders of the Lost Kek dataset](https://arxiv.org/abs/2001.07487), which contains over 3.3 million threads and 134.5 million posts from /pol/. Each dataset entry is a JSON file representing a /pol/ thread.

The dataset is preprocessed by reformatting each entry into the following structure:

```text
---(post start) No.
Content
-----(thread end)
(2 new lines after a thread)
```

Here's an example thread in the reformatted style:

```text
--- 943264
Hi /pol/
--- 943265
>> 943264
Hi anon
------


```

The preprocessed dataset is then tokenized using the tokenizer from [GPT-NeoX](https://arxiv.org/abs/2204.06745) and stored as numpy memmap files with uint16 dtype. These steps reduce the dataset size from 106 GB to 11 GB, making distribution much easier. You can generate the memmap file using [generate dataset.ipynb](generate%20dataset.ipynb), or you can download the pre-generated memmap:

| Raw Text Download             | Num. of Char. | Tokenized Download             | Num. of Tokens |
|-------------------------------|---------------|--------------------------------|----------------|
| [Download][raw text download] | 21B           | [Download][tokenized download] | 6B             |

[raw text download]: https://archive.org/details/mamba_4chan_dataset_raw
[tokenized download]: https://archive.org/details/mamba_4chan_dataset

## Fine-tuned Models

We provide the following fine-tuned models, each trained for one epochs on the tokenized dataset using a single RTX A6000 with a context size of 2048 tokens. Mixed precision (bf16) was used for training, while the model weights were stored in fp32. We will release more models and improved versions as opportunities arise.

| Name             | Model Dim. | Num. of Layers | Batch Size | Gradient Acc. | Download                  | Fine-tuning Log |
|------------------|------------|----------------|------------|---------------|---------------------------|-----------------|
| Mamba 4chan 130M | 768        | 24             | 20         | 60            | [Download][130M download] | [log][130M log] |
| Mamba 4chan 370M | 1024       | 48             | 12         | 100           | [Download][370M download] | [log][370M log] |

[130M download]: https://archive.org/details/mamba_4chan_130m
[130M log]: https://wandb.ai/catalpa/Mamba%204chan%20130m
[370m download]: https://archive.org/details/mamba_4chan_370m
[370M log]: https://wandb.ai/catalpa/Mamba%204chan%20370m

## Training and Inferencing

We provide [mamba 4chan train.ipynb](mamba%204chan%20train.ipynb), which contains all the necessary code to train a Mamba 4chan model and log the training progress. The logged parameters can be modified in [model.py](model.py).

The base model's hyperparameters are stored in [model_config.py](model_config.py), and you can adjust them as needed. When further training our model, note that all hyperparameters are saved directly in the model file. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint). The same applies to inferencing, as PyTorch Lightning automatically handles all parameters when loading our model.

Here's a sample code snippet to perform inferencing with Mamba 4chan:

```python
from transformers import AutoTokenizer

from model import mamba_4chan

model = mamba_4chan.load_from_checkpoint("path_to.ckpt")
model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
text = "--- 94326400\nHi /pol/, lets have a thread about".rstrip()
pred = model.generate_text(tokenizer, text, 256)
```

You can also use this [colab notebook](https://colab.research.google.com/drive/1AEebKVl0gBOg75G8kYwIPUZdlK0kRTK4) for a quick demo.

## Credits

Our work builds upon the remarkable achievement of [Mamba](https://arxiv.org/abs/2312.00752) <3.

Some code for dataset preprocessing is taken from [here](https://github.com/yk/gpt-4chan-public/blob/master/src/process_data.py).
