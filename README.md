# DegFormer for Remaining Useful Life Estimation
### Project setup
```
pip install -r requirements.txt
```

### Train & Evaluate
```
python ./train_script/lightning_cmapss_FD001.py --dataset-root /your/path/to/CMAPSS --sub-dataset FD001
```

### Test & Evaluate with trained models
```
python ./test_script/FD001_Test.py --dataset-root /your/path/to/CMAPSS --sub-dataset FD001
```