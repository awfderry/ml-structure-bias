# Protein Sequence Design

To download the dataset from Zenodo, run the `download_data.sh` script.
    

To train the design GVP model, run the following:

    python train.py --train --name design_mixed --cath-data ../data/design_zenodo_data/design_chains.jsonl --cath-splits ../data/design_zenodo_data/design_mixed.json

To evaluate the trained model on the test set, use the same run name with the `--test` flag.

    python train.py --test --name design_mixed --cath-data ../data/design_zenodo_data/design_chains.jsonl --cath-splits ../data/design_zenodo_data/design_mixed.json