# Catalytic Residue Prediction Task

## Workflow
1. `conda crate --name cat_env --file cat_env.txt` to create the conda environment
2. `conda activate cat_env` to activate the conda environment
3. `./download_data.sh` to download necessary data from Zenodo
4. `./train_xray.sh` and `./train_mixed.sh` to train one model on only X-ray crystallography data, and another model on all three types of data, respectively
5. `python test.py` to run predictions on both trained models for both X-ray and NMR test data
6. `python evaluate_ec_pred.py [TEST SPLIT] [TRAIN SPLIT]` to evaluate the EC number prediction of the model trained on [TRAIN SPLIT] for the [TEST SPLIT] type of test data (e.g. `python evaluate_ec_pred.py test_nmr train_mixed`)
7. `python evaluate_cat_res_pred.py [TEST SPLIT] [TRAIN SPLIT]` to evaluate the catalytic residue prediction of the model trained on [TRAIN SPLIT] for the [TEST SPLIT] type of test data (e.g. `python evaluate_cat_res_pred.py test_nmr train_mixed`)
8. `python make_figures.py` to recreate figures from the paper
9. `conda deactivate` to deactivate the conda environment
