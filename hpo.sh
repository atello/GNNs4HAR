#!/bin/bash
ray start --head --num-cpus 4 --num-gpus 1
sleep 10

python3 hyperparameters_optimization.py --ds_name=PAMAP2 --ds_variant=corrcoef_all2_interpolate --model_name=graphconv --epochs=5 --batch_size=64 --num_layers=3 --input_dim=512 --out_dim=12
ray stop