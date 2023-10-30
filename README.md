## STEREO-GCN: Uncovering Sparse Temporal Gene Regulatory Networks using Evolving Graph Convolutional Networks

To prepare the data, see

- notebooks/pseudotime-ordering.ipynb
- notebooks/polynomial-fit.ipynb

First install 'stereo' through pip (make sure you are in the directory
containing `setup.py`)

```bash
pip install -e .
```

We use `hydra` to parse `yaml` files containing model params. To train a
model, use the following format

```python
python main.py dataset=$DATASET_NAME module=$MODULE_NAME module_conf=$MODULE_CONF_NAME
```

For `$DATASET_NAME` pick any dataset under `configs/dataset`. Currently
supported modules are:

- STEREO_GCN_Module
- TVDBN
- TemporalDeepAutoreg_Module
- TVGL

The config file specifying the parameters for that module should also be
specified in `$MODULE_CONF_NAME`. This takes any config under
`configs/module_conf`.

For example, to run the TF selection step of STEREO-GCN on the PBMC COVID vaccine data, run

```python
python main.py dataset=GRN-PBMC module=STEREO_GCN_Module module_conf=stereo-hier-prox
```

Next, construct a file with sources IDs to be used in the full-training
step (examples under `data/sources`) and specify the path to this file

```python
python main.py dataset=GRN-PBMC module=STEREO_GCN_Module module_conf=stereo-full \
     sources_ckpt=data/sources/PBMC.txt
```

It is recommended to install `wandb` for logging.
