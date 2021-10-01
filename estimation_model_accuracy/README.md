# Estimation of Model Accuracy

To download the dataset from Zenodo, run the `download_data.sh` script.

To train and evaluate the EMA GVP model, run the following:

    python run_mqa.py --data_dir ../data/ema_zenodo_data/casp_mixed_lmdb/data --batch_size 12 --mode test --name casp_mixed --num_epochs 25

Output predictions will be saved to the directory specified by `--log_dir` (defaults to `logs/{args.name}_test`).