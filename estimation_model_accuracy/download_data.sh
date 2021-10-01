#!/bin/bash

cd ../data
wget 'https://zenodo.org/record/5542201/files/ema_zenodo_data.tar.gz?download=1'
tar -xzvf ema_zenodo_data.tar.gz
cd -