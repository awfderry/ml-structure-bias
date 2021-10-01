#!/bin/bash

wget https://zenodo.org/record/5542201/files/enz_cat_res_zenodo_data.tar.gz
tar xvf enz_cat_res_zenodo_data.tar.gz
mv enz_cat_res_zenodo_data data
rm enz_cat_res_zenodo_data.tar.gz
