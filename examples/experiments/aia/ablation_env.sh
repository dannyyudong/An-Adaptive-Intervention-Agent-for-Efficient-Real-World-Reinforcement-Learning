# Sourced by standard AIA launchers. No changes for legacy runs.
if [[ -n "${AIA_ABLATION:-}" ]]; then
    case "$AIA_ABLATION" in
        rl_only) export LEARNED_OPTION_SCHEDULER=0 ;;
        no_recovery|no_codepolicy|fixed_rule|ours) export LEARNED_OPTION_SCHEDULER=1 ;;
        *) echo "Unknown AIA_ABLATION=$AIA_ABLATION" >&2; exit 1 ;;
    esac
    export MANUAL_OPTION_SCHEDULER=0
    if [[ "${NEW_RUN:-0}" == "1" && -z "${RUN_ID:-}" ]]; then
        export RUN_ID="${AIA_ABLATION}_$(date +%Y%m%d_%H%M%S)"
    fi
fi
