#!/usr/bin/env bash
# =============================================================================
# run_timing.sh  –  Group 70 / 3DY4  –  Timing Measurement Automation
#
# Runs project binary across all modes (0-3), all CPU frequency settings,
# and all filter-tap counts.  Collects the timing summary CSV lines printed
# to stderr, then writes a single aggregate results file ready to paste into
# Google Docs or open in a spreadsheet.
#
# Usage:
#   chmod +x run_timing.sh
#   ./run_timing.sh
#
# Requirements on the RPi:
#   • Binary already compiled:  make  (produces ./build/project)
#   • rtl_sdr installed and antenna connected  (or supply a recorded .bin file)
#   • sudo access for cpufreq-set  (or use cpufreq-info to confirm active freq)
#   • Recorded IQ data files named  data/iq_mode<N>.bin  (see note below)
#
# If you have a LIVE RTL-SDR, replace the  cat data/iq_mode${MODE}.bin  lines
# with:  rtl_sdr -s ${RF_FS} -f 97700000 -n ${SAMPLES} - 2>/dev/null
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
BINARY="./build/project"
DATA_DIR="./data"
RESULTS_DIR="./timing_results"
RESULT_FILE="${RESULTS_DIR}/timing_all_$(date +%Y%m%d_%H%M%S).csv"

mkdir -p "${RESULTS_DIR}"

# ── CPU frequency steps to test (Hz) ─────────────────────────────────────────
# Map to governor "userspace" so we can pin frequency.
# Typical RPi 4 OPPs: 600, 900, 1200, 1500, 1800 MHz
CPU_FREQS=(600000 900000 1200000 1500000)

# Corresponding filter-tap counts per the project spec
declare -A TAPS_FOR_FREQ
TAPS_FOR_FREQ[600000]=81
TAPS_FOR_FREQ[900000]=87
TAPS_FOR_FREQ[1200000]=95
TAPS_FOR_FREQ[1500000]=101

# ── Modes to test ─────────────────────────────────────────────────────────────
# Mode 0,1: mono  /  Mode 0,1: stereo  /  Mode 0,2: RDS+stereo
# For simplicity we test all combinations and let the binary flag unsupported ones.
declare -A MODE_PATH
MODE_PATH[0]="m s r"   # mode 0 supports mono, stereo, RDS
MODE_PATH[1]="m s"     # mode 1: mono + stereo (no RDS)
MODE_PATH[2]="m s r"   # mode 2 supports mono, stereo, RDS
MODE_PATH[3]="m s"     # mode 3: mono + stereo (no RDS)

# ── RF sample rates per mode (for sizing the capture) ────────────────────────
declare -A RF_FS
RF_FS[0]=2400000
RF_FS[1]=2304000
RF_FS[2]=2400000
RF_FS[3]=1800000

# Number of IQ samples to feed (200 blocks × block_size is handled by binary,
# but we supply enough raw bytes; 200 blocks × blocksize_mode0 ≈ 192M bytes)
# We use a generous value and let the binary exit after N_TIMING_BLOCKS.
IQ_BYTES=250000000   # ~250 MB – more than enough for 200 blocks in any mode

# ── Helper: set CPU frequency on all cores ────────────────────────────────────
set_cpu_freq() {
    local FREQ_HZ=$1
    echo "  [cpu] Setting all cores to ${FREQ_HZ} Hz..."
    for CPU in /sys/devices/system/cpu/cpu[0-9]*; do
        GOVERNOR_FILE="${CPU}/cpufreq/scaling_governor"
        FREQ_FILE="${CPU}/cpufreq/scaling_setspeed"
        if [ -f "${GOVERNOR_FILE}" ]; then
            echo userspace | sudo tee "${GOVERNOR_FILE}" > /dev/null
            echo "${FREQ_HZ}" | sudo tee "${FREQ_FILE}" > /dev/null
        fi
    done
    sleep 1   # allow governor to settle
    # Read back actual frequency for verification
    ACTUAL=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "unknown")
    echo "  [cpu] Actual freq: ${ACTUAL} Hz"
}

# ── Helper: restore default governor (on-demand / performance) ───────────────
restore_cpu_gov() {
    echo "  [cpu] Restoring ondemand governor..."
    for CPU in /sys/devices/system/cpu/cpu[0-9]*; do
        GOVERNOR_FILE="${CPU}/cpufreq/scaling_governor"
        if [ -f "${GOVERNOR_FILE}" ]; then
            echo ondemand | sudo tee "${GOVERNOR_FILE}" > /dev/null 2>&1 || true
        fi
    done
}

# ── Ensure IQ data files exist (or generate synthetic ones) ──────────────────
generate_iq_if_needed() {
    local MODE=$1
    local FS=${RF_FS[$MODE]}
    local OUTFILE="${DATA_DIR}/iq_mode${MODE}.bin"
    if [ ! -f "${OUTFILE}" ]; then
        echo "  [data] Generating synthetic IQ data for mode ${MODE} (${IQ_BYTES} bytes)..."
        mkdir -p "${DATA_DIR}"
        # Write random bytes simulating unsigned 8-bit IQ samples
        dd if=/dev/urandom bs=1M count=$(( IQ_BYTES / 1000000 + 1 )) 2>/dev/null \
            | head -c "${IQ_BYTES}" > "${OUTFILE}"
        echo "  [data] Written ${OUTFILE}"
    fi
}

# ── Write CSV header ──────────────────────────────────────────────────────────
echo "cpu_freq_hz,mode,path,filter_taps,block_name,total_ms,avg_ms_per_block" \
    > "${RESULT_FILE}"

echo "=========================================================="
echo "  3DY4 Group 70 – Timing Measurement Script"
echo "  Results will be written to: ${RESULT_FILE}"
echo "=========================================================="

# ── Main sweep ────────────────────────────────────────────────────────────────
for FREQ in "${CPU_FREQS[@]}"; do
    TAPS=${TAPS_FOR_FREQ[$FREQ]}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CPU Freq = ${FREQ} Hz   Filter Taps = ${TAPS}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    set_cpu_freq "${FREQ}"

    for MODE in 0 1 2 3; do
        generate_iq_if_needed "${MODE}"
        IQ_FILE="${DATA_DIR}/iq_mode${MODE}.bin"

        for PATH_FLAG in ${MODE_PATH[$MODE]}; do
            echo ""
            echo "  ── Mode=${MODE}  Path=${PATH_FLAG}  Taps=${TAPS}"

            TMPLOG=$(mktemp)

            # Run binary, pipe IQ data in, capture stderr (timing + debug)
            # stdout (PCM audio) is discarded.
            if cat "${IQ_FILE}" \
                | "${BINARY}" "${MODE}" "${PATH_FLAG}" "${TAPS}" \
                  2>"${TMPLOG}" \
                  > /dev/null; then
                :
            else
                echo "  [warn] Binary exited with non-zero status (may be normal after N blocks)"
            fi

            # Parse the CSV block between "=== TIMING SUMMARY ===" markers
            IN_SUMMARY=0
            SKIPPED_HEADER=0
            while IFS= read -r LINE; do
                if [[ "$LINE" == *"=== TIMING SUMMARY ==="* ]]; then
                    IN_SUMMARY=1
                    continue
                fi
                if [[ "$LINE" == *"=== END TIMING SUMMARY ==="* ]]; then
                    IN_SUMMARY=0
                    continue
                fi
                if [[ $IN_SUMMARY -eq 1 ]]; then
                    # Skip the informational line "Mode=... Blocks=..."
                    if [[ "$LINE" == Mode=* ]]; then continue; fi
                    # Skip the CSV header line emitted by the binary
                    if [[ $SKIPPED_HEADER -eq 0 && "$LINE" == block_name* ]]; then
                        SKIPPED_HEADER=1
                        continue
                    fi
                    # Data rows: prepend our columns
                    if [[ -n "$LINE" ]]; then
                        echo "${FREQ},${MODE},${PATH_FLAG},${TAPS},${LINE}" \
                            >> "${RESULT_FILE}"
                    fi
                fi
            done < "${TMPLOG}"

            rm -f "${TMPLOG}"
        done
    done
done

restore_cpu_gov

echo ""
echo "=========================================================="
echo "  All measurements complete."
echo "  Results file: ${RESULT_FILE}"
echo "=========================================================="

# ── Pretty-print summary to terminal ─────────────────────────────────────────
echo ""
echo "Aggregated CSV (first 40 lines):"
head -40 "${RESULT_FILE}"
