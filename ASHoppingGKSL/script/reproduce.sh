#!/usr/bin/env bash

bash /Users/michishita-su/Documents/Code/ASHopping-Lindblad/ASHoppingGKSL/script/plot_jump_heatmap_gamma_sweep.sh \                      
  --outroot ./out/nr_sweep6 \
  0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 -- \
  --nsites 6 --mu 2.4 --beta 10.0 --tau-c 0.002 --bathkind bath \
  --method nr --formats pdf --matrix-formats jld2 --components abs

julia ./ASHoppingGKSL/script/plot_current_occupation_profile.jl /Users/michishita-su/Documents/Code/ASHopping-Lindblad/out/nr_sweep6