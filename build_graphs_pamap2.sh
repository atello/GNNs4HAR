
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=no_edges --fillnan=interpolate
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=no_edges --fillnan=zero
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=no_edges --fillnan=dropna

python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=ensemble --fillnan=interpolate --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=ensemble --fillnan=zero --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=ensemble --fillnan=dropna --corr_threshold=0.2

python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_all --fillnan=interpolate --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_all --fillnan=zero --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_all --fillnan=dropna --corr_threshold=0.2

python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_win --fillnan=interpolate --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_win --fillnan=zero --corr_threshold=0.2
python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=corrcoef_win --fillnan=dropna --corr_threshold=0.2

#python3 build_graphs.py --ds_name=PAMAP2 --ds_variant=gaussian_win --corr_threshold=0.0