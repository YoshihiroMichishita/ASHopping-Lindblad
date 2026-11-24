#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
outroot="$project_dir/../out/nr_sweep6"

bash "$script_dir/plot_jump_heatmap_gamma_sweep.sh" \
  --outroot "$outroot" \
  0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 -- \
  --nsites 6 --mu 2.4 --beta 10.0 --tau-c 0.002 --bathkind bath \
  --method nr --formats pdf --matrix-formats jld2 --components abs

julia --project="$project_dir" \
  "$script_dir/plot_current_occupation_profile.jl" "$outroot/"