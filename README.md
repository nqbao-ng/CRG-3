- IEMOCAP

Run from the workspace root. The default validation ratio is `0.1`, and the
checkpoint is selected by validation weighted F1.

```powershell
python .\CRG-3\run.py --dataset IEMOCAP --data_path .\data\iemocap_multi_features.pkl --out_dir .\CRG-3\outputs\iemocap_val
```

The classifier concatenates the three CRG outputs. Add
`--attention_history_scope intra` to restrict Same/Diff history to the target modality.

- MELD

'python run.py --dataset MELD --data_path .\data\meld_multi_features.pkl --validation_split native --valid_ratio 0.1 --out_dir .\CRG-3\outputs\meld'
