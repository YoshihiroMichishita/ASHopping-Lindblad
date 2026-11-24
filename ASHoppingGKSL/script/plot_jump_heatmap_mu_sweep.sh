#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

outroot="${project_dir}/out/jump_heatmap_mu_sweep"
#mus=(0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4)
#mus=(0.0 0.6 1.2 1.8 2.4)
mus=(0.1)
plot_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot)
      [[ $# -ge 2 ]] || { echo "Missing value for --outroot" >&2; exit 1; }
      outroot="$2"
      shift 2
      ;;
    --)
      shift
      plot_args=("$@")
      break
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Usage: $0 [--outroot DIR] -- [plot_jump_heatmap args]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$outroot"

for mu in "${mus[@]}"; do
  mu_dir="${outroot}/mu_${mu}"
  echo "Running plot_jump_heatmap.jl for mu=${mu}, Gamma=0.1 -> ${mu_dir}"
  julia --project="${project_dir}" "${script_dir}/plot_jump_heatmap.jl" \
    "${plot_args[@]}" \
    --Gamma 50.0 \
    --mu "${mu}" \
    --outdir "${mu_dir}"
done
