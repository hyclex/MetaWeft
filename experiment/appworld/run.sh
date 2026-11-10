#!/usr/bin/env bash
set -euo pipefail
entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
setting=${1:-}
case "$setting" in
  base|direct|retrieval|metaweft_v1) ;;
  *) echo 'Usage: bash MetaWeft/experiment/appworld/run.sh {base|direct|retrieval|metaweft_v1} [options] (retrieval requires --preset); V1 design: metaweft_v1.variant.json' >&2; exit 2 ;;
esac
shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONPYCACHEPREFIX
exec python3 -B "$entry_dir/$setting.py" "$@"
