#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

outroot="${project_dir}/out/jump_heatmap_gamma_sweep"
gammas=()
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
      gammas+=("$1")
      shift
      ;;
  esac
done

[[ ${#gammas[@]} -gt 0 ]] || { echo "Usage: $0 [--outroot DIR] GAMMA [GAMMA ...] -- [plot_jump_heatmap args]" >&2; exit 1; }

mkdir -p "$outroot"

for gamma in "${gammas[@]}"; do
  gamma_dir="${outroot}/Gamma_${gamma}"
  echo "Running plot_jump_heatmap.jl for Gamma=${gamma} -> ${gamma_dir}"
  julia --project="${project_dir}" "${script_dir}/plot_jump_heatmap.jl" \
    --Gamma "${gamma}" \
    --outdir "${gamma_dir}" \
    "${plot_args[@]}"
done
