# Remaining Useful Life Estimation Based on Improved Attention Mechanism
we propose the Degradation-Augmented RUL (DA-RUL) model, which incorporates distance-based similarity measures into the self-attention mechanism to emphasize geometrically significant variations. By augmenting these sequences, the model is better able to focus on critical degradation events, improving its overall predictive accuracy. Subsequently, the sLSTM model is employed to capture both temporal and local dependencies within the data. The model's performance is evaluated on NASA's jet engine run-to-failure datasets and XJTU-SY bearning datasets, demonstrating that DA-RUL not only enhances RUL estimation accuracy but also adapts effectively to diverse operating conditions with low computational complexity.

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

We compare our model with 14 baselines, including Dual_Mixer, FC-STGNN, CTNet, etc. **DA-RUL achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

## Data
The `--dataset-root` should be updated according to your data folder. 

The data can be downloaded here.

NASA CMAPSS Jet Engine Simulated Data: ([https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data))

XJTU-SY bearing Datasets: (https://biaowang.tech/xjtu-sy-bearing-datasets/)

### Train & Evaluate
```
python ./train_script/lightning_cmapss_FD001.py --dataset-root /your/path/to/CMAPSS --sub-dataset FD001
python ./train_script/xjtu_bearing.py --dataset-root /your/path/to/XJTU --sub-dataset 37
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
