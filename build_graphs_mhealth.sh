#!/bin/bash
# ************************ Correlation Coefficient Graphs ************************
# all data adjacency matrix

python3 build_graphs.py --ds_name=MHEALTH --ds_variant=no_edges
python3 build_graphs.py --ds_name=MHEALTH --ds_variant=ensemble --corr_threshold=0.2

python3 build_graphs.py --ds_name=MHEALTH --ds_variant=corrcoef_all --corr_threshold=0.2
python3 build_graphs.py --ds_name=MHEALTH --ds_variant=corrcoef_win --corr_threshold=0.2
python3 build_graphs.py --ds_name=MHEALTH --ds_variant=gaussian_win --corr_threshold=0.0
