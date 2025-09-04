# Degradation-Augmented Transformer
Degradation-Augmented Transformer: Degradation-Augmented Transformer for Enhanced RUL Estimation using Multi-fault Mode Sensor Data

<p align="center">
  <img src=".\pics\Architecture.png" height = "350" alt="" align=center />
</p>

## Requirements

- Python 3.8.19
- numpy == 1.24.4
- pandas == 1.5.3
- scikit_learn == 0.24.0
- torch==1.8.2+cu111
- pytorch_lightning==1.1.4


Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. **Generally,  Anomaly-Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

## Data
The `--dataset-root` should be updated according to your data folder. 

The data can be downloaded here.

NASA CMAPSS Jet Engine Simulated Data: (https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data)
### Train & Evaluate
```
python ./train_script/lightning_cmapss_FD001.py --dataset-root /your/path/to/CMAPSS --sub-dataset FD001
```

### Test & Evaluate with trained models
```
python ./test_script/FD001_Test.py --dataset-root /your/path/to/CMAPSS --sub-dataset FD001
```
## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.

- lucidrains (https://github.com/lucidrains/iTransformer)
- LazyLZ (https://github.com/LazyLZ/multi-head-attention-for-rul-estimation)
- muditbhargava66(https://github.com/muditbhargava66/PyxLSTM)
- 
