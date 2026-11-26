#!/usr/bin/env bash
set -euo pipefail
entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
setting=${1:-}
case "$setting" in
  base|direct|retrieval|efficiency_1_emb3072|metaweft_v1|metaweft_v2|contract_check|observation_dict) ;;
  *) echo 'Usage: bash MetaWeft/experiment/appworld/run.sh {base|direct|retrieval|efficiency_1_emb3072|metaweft_v1|metaweft_v2|contract_check|observation_dict} [options] (retrieval requires --preset); each entry has an adjacent .variant.json design' >&2; exit 2 ;;
esac
shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONPYCACHEPREFIX
exec python3 -B "$entry_dir/$setting.py" "$@"
