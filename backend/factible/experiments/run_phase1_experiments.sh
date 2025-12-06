#!/bin/bash

################################################################################
# Phase 1: Global Quantitative Analysis - Experiment Runner
#
# This script runs all recommended experiments for Phase 1 of the thesis.
# It includes OFAT analysis, strategic comparison, and topic diversity.
#
# Usage:
#   ./run_phase1_experiments.sh                    # Run all experiments
#   ./run_phase1_experiments.sh --ofat-only        # Run only OFAT experiments
#   ./run_phase1_experiments.sh --strategic-only   # Run only strategic experiments
#   ./run_phase1_experiments.sh --topic-only       # Run only topic diversity
#
# Estimated time: 4-5 hours for all experiments
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Go up two levels: experiments -> factible -> backend
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/phase1_run_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Parse command line arguments
RUN_OFAT=true
RUN_STRATEGIC=true
RUN_TOPIC=true
RESUME_SUBDIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ofat-only)
            RUN_STRATEGIC=false
            RUN_TOPIC=false
            shift
            ;;
        --strategic-only)
            RUN_OFAT=false
            RUN_TOPIC=false
            shift
            ;;
        --topic-only)
            RUN_OFAT=false
            RUN_STRATEGIC=false
            shift
            ;;
        --resume)
            RESUME_SUBDIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ofat-only       Run only OFAT experiments (vary_claims, vary_queries, vary_results)"
            echo "  --strategic-only  Run only strategic experiments (minimal, deep, broad)"
            echo "  --topic-only      Run only topic diversity experiments"
            echo "  --resume <subdir> Resume an interrupted run (e.g., vary_claims_20251206_191228)"
            echo "  --help            Show this help message"
            echo ""
            echo "If no options are provided, all experiments will be run."
            echo ""
            echo "Examples:"
            echo "  $0 --ofat-only"
            echo "  $0 --resume vary_claims_20251206_191228 --ofat-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}║${NC} $1" | tee -a "$LOG_FILE"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Change to backend directory
cd "$BACKEND_DIR"

# Print header
clear
echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Phase 1: Global Quantitative Analysis                      ║
║   Automated Experiment Runner                                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log "Starting Phase 1 experiments"
log "Log file: $LOG_FILE"
log "Working directory: $BACKEND_DIR"
echo ""

# Calculate what will run
TOTAL_EXPERIMENTS=0
ESTIMATED_TIME=0

if [ "$RUN_OFAT" = true ]; then
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 3))  # 3 OFAT experiments
    ESTIMATED_TIME=$((ESTIMATED_TIME + 180))  # ~3 hours
fi

if [ "$RUN_STRATEGIC" = true ]; then
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 3))  # 3 strategic experiments
    ESTIMATED_TIME=$((ESTIMATED_TIME + 45))   # ~45 minutes
fi

if [ "$RUN_TOPIC" = true ]; then
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 4))  # 4 individual videos
    ESTIMATED_TIME=$((ESTIMATED_TIME + 15))   # ~15 minutes
fi

log "Configuration:"
log "  - OFAT experiments: $([ "$RUN_OFAT" = true ] && echo "YES" || echo "NO")"
log "  - Strategic experiments: $([ "$RUN_STRATEGIC" = true ] && echo "YES" || echo "NO")"
log "  - Topic diversity: $([ "$RUN_TOPIC" = true ] && echo "YES" || echo "NO")"
log "  - Total experiments: $TOTAL_EXPERIMENTS"
log "  - Estimated time: ~$ESTIMATED_TIME minutes (~$((ESTIMATED_TIME/60)) hours)"

# Check resume directory if provided
if [ -n "$RESUME_SUBDIR" ]; then
    RESUME_PATH="$BACKEND_DIR/factible/experiments/runs/$RESUME_SUBDIR"
    if [ -d "$RESUME_PATH" ]; then
        EXISTING_RUNS=$(find "$RESUME_PATH" -mindepth 1 -maxdepth 1 -type d | wc -l)
        log "  - Resume mode: YES"
        log "  - Resume directory: $RESUME_SUBDIR"
        log "  - Existing runs: $EXISTING_RUNS (will be skipped)"
    else
        log_error "Resume directory does not exist: $RESUME_PATH"
        log_error "Please check the directory name and try again"
        exit 1
    fi
else
    log "  - Resume mode: NO (will create new directories)"
fi
echo ""

# Confirm before starting
echo -e "${YELLOW}This will run $TOTAL_EXPERIMENTS experiments.${NC}"
echo -e "${YELLOW}Estimated completion time: ~$ESTIMATED_TIME minutes${NC}"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "Execution cancelled by user"
    exit 0
fi

# Track progress
COMPLETED_EXPERIMENTS=0
START_TIME=$(date +%s)

# Function to run experiment and track progress
run_experiment() {
    local experiment_name=$1
    local description=$2
    local estimated_minutes=$3

    COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))

    # Use resume directory if provided, otherwise generate new one
    local subdir
    if [ -n "$RESUME_SUBDIR" ]; then
        subdir="$RESUME_SUBDIR"
        log_section "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] $description (RESUMING)"
        log "Resuming in existing directory: $RESUME_SUBDIR"
    else
        subdir="${experiment_name}_$(date +%Y%m%d_%H%M%S)"
        log_section "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] $description"
    fi

    log "Running: factible-experiments run --experiment $experiment_name --runs-subdir $subdir"
    log "Runs will be saved to: factible/experiments/runs/$subdir/"
    log "Estimated time: ~$estimated_minutes minutes"

    local exp_start=$(date +%s)

    if uv run factible-experiments run --experiment "$experiment_name" --runs-subdir "$subdir" 2>&1 | tee -a "$LOG_FILE"; then
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        log_success "Completed in $((exp_duration/60)) minutes $((exp_duration%60)) seconds"
        log "Results saved in: runs/$subdir/"
    else
        log_error "Failed to complete experiment: $experiment_name"
        log_error "Check log file for details: $LOG_FILE"
        return 1
    fi

    echo ""
}

# Function to run baseline on specific video
run_baseline_video() {
    local video_id=$1
    local description=$2

    COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))

    # Use resume directory if provided, otherwise generate new one
    local subdir
    if [ -n "$RESUME_SUBDIR" ]; then
        subdir="$RESUME_SUBDIR"
        log_section "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] $description (RESUMING)"
        log "Resuming in existing directory: $RESUME_SUBDIR"
    else
        subdir="baseline_${video_id}_$(date +%Y%m%d_%H%M%S)"
        log_section "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] $description"
    fi

    log "Running: factible-experiments run --experiment baseline --video $video_id --runs-subdir $subdir"
    log "Runs will be saved to: factible/experiments/runs/$subdir/"

    local exp_start=$(date +%s)

    if uv run factible-experiments run --experiment baseline --video "$video_id" --runs-subdir "$subdir" 2>&1 | tee -a "$LOG_FILE"; then
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        log_success "Completed in $((exp_duration/60)) minutes $((exp_duration%60)) seconds"
        log "Results saved in: runs/$subdir/"
    else
        log_error "Failed to complete baseline for video: $video_id"
        log_error "Check log file for details: $LOG_FILE"
        return 1
    fi

    echo ""
}

################################################################################
# TIER 1: OFAT EXPERIMENTS (CRITICAL)
################################################################################

if [ "$RUN_OFAT" = true ]; then
    log_section "TIER 1: OFAT Sensitivity Analysis (Critical for Thesis)"
    log "Running OFAT experiments: vary_claims, vary_queries, vary_results"
    log "This will unlock Graph 06: OFAT Sensitivity Analysis"
    echo ""

    # Run each OFAT experiment sequentially
    run_experiment "vary_claims" "OFAT: Varying max_claims (1,3,5,7,10)" 60 || exit 1
    run_experiment "vary_queries" "OFAT: Varying max_queries_per_claim (1,2,3,4,5)" 60 || exit 1
    run_experiment "vary_results" "OFAT: Varying max_results_per_query (1,2,3,5,7)" 60 || exit 1

    log_success "OFAT experiments complete!"
    log "Total runs completed: $(ls -1 factible/experiments/runs/ | wc -l)"
fi

################################################################################
# TIER 1: STRATEGIC COMPARISON
################################################################################

if [ "$RUN_STRATEGIC" = true ]; then
    log_section "TIER 1: Strategic Configuration Comparison"
    log "Running strategic experiments: minimal, deep, broad"
    log "This will unlock Graph 07: Strategy Comparison"
    echo ""

    # Run each strategic experiment sequentially
    run_experiment "minimal" "Strategic: Minimal (fast & cheap)" 15 || exit 1
    run_experiment "deep" "Strategic: Deep (thorough analysis)" 15 || exit 1
    run_experiment "broad" "Strategic: Broad (wide coverage)" 15 || exit 1

    log_success "Strategic experiments complete!"
    log "Total runs completed: $(ls -1 factible/experiments/runs/ | wc -l)"
fi

################################################################################
# TIER 2: TOPIC DIVERSITY
################################################################################

if [ "$RUN_TOPIC" = true ]; then
    log_section "TIER 2: Topic Diversity (Generalizability)"
    log "Adding baseline runs on tech and health videos"
    log "Demonstrates system works across different content types"
    echo ""

    # Technology videos
    run_baseline_video "5g_health_doctors" "Topic: Technology (5G Health)" || exit 1
    run_baseline_video "ai_jobs_godfather" "Topic: Technology (AI Jobs)" || exit 1

    # Health videos
    run_baseline_video "emf_cell_phone_radiation" "Topic: Health (EMF Radiation)" || exit 1
    run_baseline_video "alkaline_water" "Topic: Health (Alkaline Water)" || exit 1

    log_success "Topic diversity experiments complete!"
    log "Total runs completed: $(ls -1 factible/experiments/runs/ | wc -l)"
fi

################################################################################
# COMPLETION AND ANALYSIS
################################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

log_section "All Experiments Complete!"
log_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate final analysis
log "Generating comprehensive analysis..."
echo ""

ANALYSIS_NAME="phase1_complete_$(date +%Y%m%d_%H%M%S)"

if uv run factible-experiments analyze --name "$ANALYSIS_NAME" 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Analysis complete!"
    log "Results saved to: factible/experiments/analysis/$ANALYSIS_NAME"
else
    log_error "Analysis failed. Check log file: $LOG_FILE"
    exit 1
fi

echo ""
log_section "Summary"

# Count runs
RUNS_DIR="$BACKEND_DIR/factible/experiments/runs"
TOTAL_RUNS=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

log "Experiment execution summary:"
log "  ✓ Total experiments: $COMPLETED_EXPERIMENTS"
log "  ✓ Total runs in database: $TOTAL_RUNS"
log "  ✓ Execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "  ✓ Log file: $LOG_FILE"
log ""
log "Next steps:"
log "  1. Review analysis results:"
log "     cd factible/experiments/analysis/$ANALYSIS_NAME"
log "     open *.png"
log ""
log "  2. Check for graphs that are now unlocked:"
log "     - Graph 06: OFAT Sensitivity (if OFAT was run)"
log "     - Graph 07: Strategy Comparison (if strategic was run)"
log "     - Graph 10: Evidence Quality (enhanced with topic diversity)"
log ""
log "  3. Start writing thesis sections based on results"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║   Phase 1 Experiments Complete! 🎉                            ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
