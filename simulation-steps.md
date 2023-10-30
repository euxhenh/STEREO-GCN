### Build dataset and save
```python
python store_simulation.py --config-name=simulated-data-max
```

### Train hier-prox
```python
python main.py --config-name=simulation module=STEREO_GCN_Module module_conf=stereo-hier-prox-simulation seed=1
```

### Contruct a file `simulation.txt` with the selected sources from the previous step (examples under `data/sources`)

### Retrain using selected sources
```python
python main.py --config-name=simulation module=STEREO_GCN_Module module_conf=stereo-full-simulation sources_ckpt=data/sources/simulation.txt seed=1
```

### Train autoregressive baseline
```python
python main.py --config-name=simulation module=TemporalDeepAutoreg_Module module_conf=autoreg-simulation seed=1
```

### Train TV-DBN baseline
```python
python main.py --config-name=simulation module=TVDBN module_conf=tvdbn-simulation seed=1
```

### Train TVGL baseline
```python
python main.py --config-name=simulation module=TVGL module_conf=tvgl-simulation
```