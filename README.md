## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
    
    
## Experiments 

Id | Code | Description | Notes |  load_path |train-NLL |  train-F1 |  train-EM | train-AvNA | dev-NLL |  dev-F1 |  dev-EM | dev-AvNA | 
--- | --- | --- | --- | --- | --- | --- | --- |--- |--- |--- |--- |--- |
baseline | models.py | Baseline | No hyper-params tuning, hence dev-set used like test set. | save/train/baseline-05/best.pth.tar | 1.73 | 76.40 | 68.82 | 85.71 |3.35 | __57.31__ | 53.87 | 64.44 |
baseline_plus_1 | models.py | Baseline with larger modelsize and number of layers. `--hidden_size 150` , `--drop_prob 0.3` -  | The model appears to have a higher bias than baseline although the bigger capacity. This is due to the higher dropout probability. | save/train/baseline_plus_1-01/best.pth.tar | 02.27| 68.44 |60.75 | 80.29 | 03.22 | 56.62 | 53.25 | 63.75 |
baseline_char_embed | models.py | Baseline with character embedding. | The model comes with higher capacity and overfits after 3 epochs. It is necessary some regularization. | save/train/baseline_char_embed-04/best.pth.tar | |  | |  | 05.27 | 52.19 | 52.19 | 52.14|
baseline_char_embed | models.py | Baseline with character embedding and higher dropout (__0.3__ vs. 0.2). `--drop_prob 0.3` |  | save/train/baseline_char_embed-05/best.pth.tar |01.69 | 76.95 | 69.09| 86.17 | 03.37 | __58.02__ | 54.34 | 65.33|
baseline_char_embed | models.py | Baseline with character embedding and higher dropout (__0.4__ vs. 0.2). `--drop_prob 0.4` |  | save/train/baseline_char_embed-06/best.pth.tar |02.31 | 67.67| 59.99|79.78 | 03.07 | 57.13 |53.79 | 64.68|
baseline_char_embed_bert_enc | models.py | Baseline with character embedding and higher dropout (__0.3__). `--drop_prob 0.3` and BERT-encoder `--hidden_size 256` enstead of RNN-encoder| drop out too high? | save/train/baseline_char_embed_bert_enc-06/best.pth.tar |07.80 | 33.37 | 33.37| 33.36 | 06.32 | 52.19 | 52.19 | 52.14|
baseline_char_embed_bert_enc | models.py | Baseline with character embedding, dropout (__0.2__) leaving BERT encoder dropout to 0.1 `--drop_prob 0.2` and BERT-encoder `--hidden_size 256` enstead of RNN-encoder| capacity of the network too high? | save/train/baseline_char_embed_bert_enc-07/best.pth.tar |07.78 | 33.37 | 33.37| 33.36 | 06.32 | 52.19 | 52.19 | 52.14|
baseline_char_embed_bert_enc | models.py | Baseline with character embedding, dropout (__0.2__) leaving BERT encoder dropout to 0.1 `--drop_prob 0.2` and BERT-encoder `--hidden_size 256` enstead of RNN-encoder. BERT encoder has 3 layers (instead of 6) and d_ff, the depth of feed-forward layer, is ~depth of embedding (instead of 2048) |  | save/train/baseline_char_embed_bert_enc-08/best.pth.tar |07.78 | 33.37 | 33.37| 33.36 | 06.32 | 52.19 | 52.19 | 52.14|

