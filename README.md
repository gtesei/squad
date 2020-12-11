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
baseline | models.py | Baseline | No hyper-params tuning, hence dev-set used like test set. | save/train/baseline-05/best.pth.tar | 1.73 | 76.40 | 68.82 | 85.71 |3.35 | 57.31 | 53.87 | 64.44 |
baseline_plus_1 | models.py | Baseline with larger modelsize and number of layers. `--hidden_size 150` , `--drop_prob 0.3` -  | The model appears to have a higher bias than baseline although the bigger capacity. This is due to the higher dropout probability. | save/train/baseline_plus_1-01/best.pth.tar | 02.27| 68.44 |60.75 | 80.29 | 03.22 | 56.62 | 53.25 | 63.75 |
