# DiGemo

## Requirement
Checking and installing environmental requirements

```bash
pip install -r requirements.txt
```

## Datasets
The benchmark datasets used in our paper are IEMOCAP and MELD.
Following prior work, we use pre-extracted, preprocessed features and provide download links here: 
[preprocessed features](https://drive.google.com/drive/folders/1qrlada7_F-YXgIvI5SmqEVBVxnJGNf3P?usp=drive_link).
We also acknowledge [GraphSmile](https://github.com/lijfrank/GraphSmile) for releasing their code and datasets.
Please set the dataset path in run.py accordingly.

## Run 
### IEMOCAP
```bash
python -u run.py --gpu 0 --port 1530 --dataset IEMOCAP --epochs 200 --loss_type distil \
--lr 2e-5 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 5 5 5 \
--dropout_1 0.05 --dropout_2 0.2 --gammas 1.0 0.4 1.0 --num_heads 16 --temp 3.0
```
### MELD
```bash
python -u run.py --gpu 0 --port 1530 --dataset MELD --epochs 50 --loss_type distil \
--lr 5e-6 --batch_size 16 --hidden_dim 512 --win 4 4 --heter_n_layers 2 2 2 \
--dropout_1 0.2 --dropout_2 0.3 --gammas 1.0 0.2 1.0 --num_heads 32 --temp 10.0 --l2 1e-5
```
