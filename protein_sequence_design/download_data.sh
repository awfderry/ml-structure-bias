#!/bin/bash

cd ../data
wget 'https://zenodo.org/record/5542201/files/design_zenodo_data.tar.gz?download=1'
tar -xzvf design_zenodo_data.tar.gz
cd -