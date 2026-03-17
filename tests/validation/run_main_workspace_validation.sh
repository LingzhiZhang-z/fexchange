#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="$REPO_ROOT/outputs/ybox_prb_particle_20260317"
LOG_DIR="$OUTPUT_ROOT/logs"
TASK="all"
SMOKE=0
TOOLS_DIR="$REPO_ROOT/fexchange/tools"
PARSE_AWK="$REPO_ROOT/scripts/validation/parse_legacy_input.awk"
DUMP_PY="$REPO_ROOT/scripts/validation/dump_l4_npz.py"
HEFF_TO_JMU_AWK="$REPO_ROOT/scripts/validation/heff_to_jmu.awk"
YBOX_COMPARE_AWK="$REPO_ROOT/scripts/validation/ybox_compare.awk"
PRB_CHANNELS_AWK="$REPO_ROOT/scripts/validation/prb_channels.awk"
YBOX_TOL="1.0e-3"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--task all|ybox|prb] [--smoke]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

run_repo_python() {
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

source_from_command() {
    local tmp_file
    tmp_file="$(mktemp "${TMPDIR:-/tmp}/fxe_validation_env.XXXXXX")"
    if ! "$@" >"$tmp_file"; then
        rm -f "$tmp_file"
        return 1
    fi
    # shellcheck disable=SC1090
    source "$tmp_file"
    rm -f "$tmp_file"
}

load_case_meta() {
    source_from_command awk -v repo_root="$REPO_ROOT" -f "$PARSE_AWK" "$1"
}

space_list_to_toml_array() {
    printf '[%s]\n' "$(printf '%s\n' "$1" | awk '{for (i = 1; i <= NF; i++) printf("%s%s", (i > 1 ? ", " : ""), $i)}')"
}

resolve_repo_path() {
    case "$1" in
        data/*) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
        *) printf '%s/data/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

point_env() {
    awk -v sl="$1" -v U="$2" -v ratio="$3" -v zeta="$4" -v n_ele="$5" '
        function fmt12(x) { return sprintf("%.12f", x) }
        BEGIN {
            split(sl, s, /[ \t]+/)
            U_meV = 1000.0 * U
            Jh_meV = 1000.0 * U * ratio
            if (Jh_meV == 0.0) {
                tiny = 1.0e-6
                F2 = tiny
                F4 = tiny * (s[2] / s[1])
                F6 = tiny * (s[3] / s[1])
            } else {
                F2 = s[1] * Jh_meV
                F4 = s[2] * Jh_meV
                F6 = s[3] * Jh_meV
            }
            U_txt = fmt12(U_meV)
            Jh_txt = fmt12(Jh_meV)
            F2_txt = fmt12(F2)
            F4_txt = fmt12(F4)
            F6_txt = fmt12(F6)
            r42 = (F4_txt + 0.0) / (F2_txt + 0.0)
            r62 = (F6_txt + 0.0) / (F2_txt + 0.0)
            printf("U_MEV=%s\n", U_txt)
            printf("JH_MEV=%s\n", Jh_txt)
            printf("F2=%s\n", F2_txt)
            printf("F4=%s\n", F4_txt)
            printf("F6=%s\n", F6_txt)
            printf("R42=%s\n", fmt12(r42))
            printf("R62=%s\n", fmt12(r62))
            printf("L4_REL=%s\n",
                "outputs/core/n-" n_ele "_r42-" fmt12(r42) "_r62-" fmt12(r62) "_scheme-RS/U-" U_txt "_Jh-" Jh_txt "_z-" fmt12(zeta) "/L4/data.npz")
        }'
}

write_w90_config() {
    local output_path="$1"
    local hr_path
    hr_path="$(resolve_repo_path "$FILE_HR")"
    cat >"$output_path" <<EOF
hr_path = "$hr_path"
spinor = $SPINOR
energy_unit = "eV"
n_orb_per_atom = $(space_list_to_toml_array "$NDIMS")
output = "w90"

[f_site_i]
atom = $ATOM_I
cell = $(space_list_to_toml_array "$CELL_I")

[f_site_j]
atom = $ATOM_J
cell = $(space_list_to_toml_array "$CELL_J")

[[ligands]]
atom = $LIGAND_1
cell = $(space_list_to_toml_array "$CELL_L1")

[[ligands]]
atom = $LIGAND_2
cell = $(space_list_to_toml_array "$CELL_L2")
EOF
}

write_run_input() {
    local output_path="$1"
    local run_id="$2"
    local title="$3"
    local n_ele="$4"
    local slater_ratios="$5"
    local U="$6"
    local ratio="$7"
    local zeta="$8"
    local hopping_file="$9"
    local projector_file="${10}"
    source_from_command point_env "$slater_ratios" "$U" "$ratio" "$zeta" "$n_ele"
    cat >"$output_path" <<EOF
schema_version = "fxe.run_input.v1"
standard_version = "2026-02"
run_id = "$run_id"
title = "$title"

[physics]
n_ele = $n_ele
F2 = $F2
F4 = $F4
F6 = $F6

[model]
scheme = "RS"

[sopt]
U = $U_MEV
Jh = $JH_MEV
zeta = $(awk -v z="$zeta" 'BEGIN{printf "%.12f\n", z}')

[inputs]
hopping_file = "$hopping_file"
projector_file = "$projector_file"

[paths]
output_root = "./outputs"

[runtime]
start_level = "LMSM"
end_level = "L4"
on_missing_upstream = "fail"
read_first = true

[checks]
strict_mode = true
eps_profile = "default"
EOF
}

dump_heff_to_jmu() {
    local l4_path="$1"
    local heff_txt="$2"
    local jmu_txt="$3"
    run_repo_python "$DUMP_PY" --input "$l4_path" --output "$heff_txt" > /dev/null 2>&1
    awk -f "$HEFF_TO_JMU_AWK" "$heff_txt" > "$jmu_txt"
}

family_text_match() {
    local ref="$1"
    shift
    local file
    for file in "$@"; do
        if ! cmp -s "$ref" "$file"; then
            echo "0"
            return
        fi
    done
    echo "1"
}

run_ybox_family() {
    local material="$1"
    local family="$2"
    local input_glob="$3"
    local projector_dir="$OUTPUT_ROOT/ybox/$material/projector/$family"
    local case_dir first_case first_hlocal first_case_dir family_gauge zeta case_input
    local -a case_inputs=()
    local -a hlocal_files=()
    local hlocal_match summary_file raw_file aligned_file max_aligned pass_flag U ratio

    while IFS= read -r case_input; do
        case_inputs+=("$case_input")
    done < <(find "$REPO_ROOT/data/test/$material/input" -maxdepth 1 -name "$input_glob" | sort)

    [ "${#case_inputs[@]}" -eq 0 ] && return

    first_case="${case_inputs[0]}"
    for case_input in "${case_inputs[@]}"; do
        load_case_meta "$case_input"
        case_dir="$OUTPUT_ROOT/ybox/$material/$PREFIX"
        mkdir -p "$case_dir"
        write_w90_config "$case_dir/w90_extract.toml"
        (
            cd "$case_dir"
            run_repo_python "$TOOLS_DIR/w90_extract.py" "w90_extract.toml" > "w90_extract.log" 2>&1
        )
        hlocal_files+=("$case_dir/w90_h_local.dat")
    done

    load_case_meta "$first_case"
    first_case_dir="$OUTPUT_ROOT/ybox/$material/$PREFIX"
    first_hlocal="$first_case_dir/w90_h_local.dat"
    hlocal_match="$(family_text_match "$first_hlocal" "${hlocal_files[@]}")"
    mkdir -p "$projector_dir"
    (
        cd "$projector_dir"
        run_repo_python "$TOOLS_DIR/w90_decompose_h_local.py" \
            --h_local_dat "$first_hlocal" \
            --rtol 1e-2 \
            --n_ele 13 \
            --symmetry C3v \
            --mode_q3 sin \
            --direct_local_fit \
            --write_cef_config "$projector_dir/cef_config.toml" \
            > "$projector_dir/decompose.log" 2>&1
    )
    zeta="$(sed -n 's/.*zeta = \([0-9.+-eE]*\) meV.*/\1/p' "$projector_dir/decompose.log" | head -n 1)"
    run_repo_python "$TOOLS_DIR/cef_states.py" "$projector_dir/cef_config.toml" \
        --format npz \
        --output "$projector_dir/projector.npz" \
        --kramer-name "${material}_${family}_particle" \
        --silent \
        > "$projector_dir/cef_states.log" 2>&1

    load_case_meta "$first_case"
    write_run_input "$first_case_dir/run_input.toml" "${PREFIX}_gauge" "${PREFIX}_gauge" 13 "$SLATER_RATIOS" 6.0 0.10 "$zeta" \
        "$first_case_dir/w90_t_mu.dat" "$projector_dir/projector.npz"
    (
        cd "$first_case_dir"
        run_repo_python "$REPO_ROOT/fexchange/cli.py" run "run_input.toml" > "gauge_run.log" 2>&1
    )
    dump_heff_to_jmu "$first_case_dir/$L4_REL" "$first_case_dir/gauge_heff.txt" "$first_case_dir/gauge_jmu.txt"
    source_from_command awk -v mode=best -v ref_file="$REFERENCE_DIR/result_HOLE_U6.00.dat" -v ratio=0.10 -f "$YBOX_COMPARE_AWK" "$first_case_dir/gauge_jmu.txt"
    family_gauge="$GAUGE_LABEL"

    summary_file="$OUTPUT_ROOT/ybox/${material}_${family}_summary.tsv"
    printf "material\tcase\tfamily\tgauge\tzeta_meV\thlocal_text_match\tmax_aligned_err\tpass\n" > "$summary_file"

    for case_input in "${case_inputs[@]}"; do
        load_case_meta "$case_input"
        case_dir="$OUTPUT_ROOT/ybox/$material/$PREFIX"
        raw_file="$case_dir/result_PARTICLE_RAW.dat"
        aligned_file="$case_dir/result_PARTICLE_GAUGEALIGNED.dat"
        printf "#U JH ratio Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz error\n" > "$raw_file"
        printf "#U JH ratio Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz error\n" > "$aligned_file"
        max_aligned="0.0"

        if [ "$SMOKE" -eq 1 ]; then
            U_LIST="6.0"
            RATIO_LIST="0.10"
        fi

        for U in $U_LIST; do
            for ratio in $RATIO_LIST; do
                write_run_input "$case_dir/run_input.toml" "${PREFIX}_U${U}_r${ratio}" "${PREFIX}_U${U}_r${ratio}" 13 "$SLATER_RATIOS" "$U" "$ratio" "$zeta" \
                    "$case_dir/w90_t_mu.dat" "$projector_dir/projector.npz"
                (
                    cd "$case_dir"
                    run_repo_python "$REPO_ROOT/fexchange/cli.py" run "run_input.toml" >> "cli.log" 2>&1
                )
                dump_heff_to_jmu "$case_dir/$L4_REL" "$case_dir/heff.txt" "$case_dir/jmu.txt"
                source_from_command awk -v mode=apply -v ref_file="$REFERENCE_DIR/result_HOLE_U$(printf "%.2f" "$U").dat" -v ratio="$ratio" -v gauge="$family_gauge" -v U="$U" -f "$YBOX_COMPARE_AWK" "$case_dir/jmu.txt"
                printf "%s\n" "$RAW_ROW" >> "$raw_file"
                printf "%s\n" "$ALIGNED_ROW" >> "$aligned_file"
                max_aligned="$(awk -v a="$max_aligned" -v b="$ALIGNED_ERR" 'BEGIN{print (a > b ? a : b)}')"
            done
        done

        pass_flag="$(awk -v err="$max_aligned" -v tol="$YBOX_TOL" 'BEGIN{print (err <= tol ? 1 : 0)}')"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$material" "$PREFIX" "$family" "$family_gauge" "$zeta" "$hlocal_match" "$max_aligned" "$pass_flag" \
            >> "$summary_file"
    done
}

run_ybox() {
    local log_file="$LOG_DIR/ybox.log"
    : > "$log_file"
    if [ "$SMOKE" -eq 1 ]; then
        run_ybox_family "YbOBr" "1st" "soc_1st_bond1.dat" >> "$log_file" 2>&1
        return
    fi
    run_ybox_family "YbOBr" "1st" "soc_1st_*.dat" >> "$log_file" 2>&1
    run_ybox_family "YbOBr" "2nd" "soc_2nd_*.dat" >> "$log_file" 2>&1
    run_ybox_family "YbOCl" "1st" "soc_1st_*.dat" >> "$log_file" 2>&1
    run_ybox_family "YbOCl" "2nd" "soc_2nd_*.dat" >> "$log_file" 2>&1
    run_ybox_family "YbOF" "1st" "soc_1st_*.dat" >> "$log_file" 2>&1
    run_ybox_family "YbOF" "2nd" "soc_2nd_*.dat" >> "$log_file" 2>&1
}

write_prb_projector() {
    local output_file="$1"
    cat >"$output_file" <<EOF
0+0j -0.408248290464+0j
0.912870929175+0j 0+0j
0+0j 0+0j
0+0j 0+0j
0+0j 0.912870929175+0j
-0.408248290464+0j 0+0j
EOF
}

write_prb_oh_j52_projector() {
    local output_file="$1"
    local log_file="$2"
    run_repo_python "$TOOLS_DIR/cef_states.py" \
        --point-group Oh \
        --J 2.5 \
        --B4 1.0 \
        --format npz \
        --output "$output_file" \
        --kramer-name prb_oh_j52 \
        --silent \
        > "$log_file" 2>&1
}

plot_prb() {
    local prb_root="$1"
    local output_png="$2"
    gnuplot <<EOF
set terminal pngcairo size 1280,2100 enhanced font "Helvetica,18"
set output "$output_png"
set multiplot layout 3,1 rowsfirst
set key off
set grid lc rgb "#cccccc" lw 0.8
set xrange [0:0.20]
set yrange [-0.5:3.0]
set style line 1 lc rgb "#4e8b3d" lw 2
set style line 2 lc rgb "#e04b3f" lw 2
set style line 3 lc rgb "#f08a24" lw 2
set style line 4 lc rgb "#5566cc" lw 2
set style line 11 lc rgb "#4e8b3d" lw 2 dashtype 2
set style line 12 lc rgb "#e04b3f" lw 2 dashtype 2
set style line 13 lc rgb "#f08a24" lw 2 dashtype 2
set style line 14 lc rgb "#5566cc" lw 2 dashtype 2
set style line 21 lc rgb "#4e8b3d" lw 2 dashtype 3
set style line 22 lc rgb "#e04b3f" lw 2 dashtype 3
set style line 23 lc rgb "#f08a24" lw 2 dashtype 3
set style line 24 lc rgb "#5566cc" lw 2 dashtype 3
set style line 31 lc rgb "#4e8b3d" lw 2 dashtype 4
set style line 32 lc rgb "#e04b3f" lw 2 dashtype 4
set style line 33 lc rgb "#f08a24" lw 2 dashtype 4
set style line 34 lc rgb "#5566cc" lw 2 dashtype 4

set title "K_{2}PrO_{3}" noenhanced
set ylabel "COUPLING CONST. (meV)"
plot \
    "$prb_root/K/prb_curves_U2.00.dat" using 1:2 with lines ls 1, \
    "$prb_root/K/prb_curves_U2.00.dat" using 1:3 with lines ls 2, \
    "$prb_root/K/prb_curves_U2.00.dat" using 1:4 with lines ls 3, \
    "$prb_root/K/prb_curves_U2.00.dat" using 1:5 with lines ls 4, \
    "$prb_root/K/prb_curves_U3.00.dat" using 1:(1.5*\$2) with lines ls 11, \
    "$prb_root/K/prb_curves_U3.00.dat" using 1:(1.5*\$3) with lines ls 12, \
    "$prb_root/K/prb_curves_U3.00.dat" using 1:(1.5*\$4) with lines ls 13, \
    "$prb_root/K/prb_curves_U3.00.dat" using 1:(1.5*\$5) with lines ls 14, \
    "$prb_root/K/prb_curves_U4.00.dat" using 1:(2.0*\$2) with lines ls 21, \
    "$prb_root/K/prb_curves_U4.00.dat" using 1:(2.0*\$3) with lines ls 22, \
    "$prb_root/K/prb_curves_U4.00.dat" using 1:(2.0*\$4) with lines ls 23, \
    "$prb_root/K/prb_curves_U4.00.dat" using 1:(2.0*\$5) with lines ls 24, \
    "$prb_root/K/prb_curves_U6.00.dat" using 1:(3.0*\$2) with lines ls 31, \
    "$prb_root/K/prb_curves_U6.00.dat" using 1:(3.0*\$3) with lines ls 32, \
    "$prb_root/K/prb_curves_U6.00.dat" using 1:(3.0*\$4) with lines ls 33, \
    "$prb_root/K/prb_curves_U6.00.dat" using 1:(3.0*\$5) with lines ls 34

set title "Rb_{2}PrO_{3}" noenhanced
plot \
    "$prb_root/Rb/prb_curves_U2.00.dat" using 1:2 with lines ls 1, \
    "$prb_root/Rb/prb_curves_U2.00.dat" using 1:3 with lines ls 2, \
    "$prb_root/Rb/prb_curves_U2.00.dat" using 1:4 with lines ls 3, \
    "$prb_root/Rb/prb_curves_U2.00.dat" using 1:5 with lines ls 4, \
    "$prb_root/Rb/prb_curves_U3.00.dat" using 1:(1.5*\$2) with lines ls 11, \
    "$prb_root/Rb/prb_curves_U3.00.dat" using 1:(1.5*\$3) with lines ls 12, \
    "$prb_root/Rb/prb_curves_U3.00.dat" using 1:(1.5*\$4) with lines ls 13, \
    "$prb_root/Rb/prb_curves_U3.00.dat" using 1:(1.5*\$5) with lines ls 14, \
    "$prb_root/Rb/prb_curves_U4.00.dat" using 1:(2.0*\$2) with lines ls 21, \
    "$prb_root/Rb/prb_curves_U4.00.dat" using 1:(2.0*\$3) with lines ls 22, \
    "$prb_root/Rb/prb_curves_U4.00.dat" using 1:(2.0*\$4) with lines ls 23, \
    "$prb_root/Rb/prb_curves_U4.00.dat" using 1:(2.0*\$5) with lines ls 24, \
    "$prb_root/Rb/prb_curves_U6.00.dat" using 1:(3.0*\$2) with lines ls 31, \
    "$prb_root/Rb/prb_curves_U6.00.dat" using 1:(3.0*\$3) with lines ls 32, \
    "$prb_root/Rb/prb_curves_U6.00.dat" using 1:(3.0*\$4) with lines ls 33, \
    "$prb_root/Rb/prb_curves_U6.00.dat" using 1:(3.0*\$5) with lines ls 34

set title "Cs_{2}PrO_{3}" noenhanced
set xlabel "J_H/U"
plot \
    "$prb_root/Cs/prb_curves_U2.00.dat" using 1:2 with lines ls 1, \
    "$prb_root/Cs/prb_curves_U2.00.dat" using 1:3 with lines ls 2, \
    "$prb_root/Cs/prb_curves_U2.00.dat" using 1:4 with lines ls 3, \
    "$prb_root/Cs/prb_curves_U2.00.dat" using 1:5 with lines ls 4, \
    "$prb_root/Cs/prb_curves_U3.00.dat" using 1:(1.5*\$2) with lines ls 11, \
    "$prb_root/Cs/prb_curves_U3.00.dat" using 1:(1.5*\$3) with lines ls 12, \
    "$prb_root/Cs/prb_curves_U3.00.dat" using 1:(1.5*\$4) with lines ls 13, \
    "$prb_root/Cs/prb_curves_U3.00.dat" using 1:(1.5*\$5) with lines ls 14, \
    "$prb_root/Cs/prb_curves_U4.00.dat" using 1:(2.0*\$2) with lines ls 21, \
    "$prb_root/Cs/prb_curves_U4.00.dat" using 1:(2.0*\$3) with lines ls 22, \
    "$prb_root/Cs/prb_curves_U4.00.dat" using 1:(2.0*\$4) with lines ls 23, \
    "$prb_root/Cs/prb_curves_U4.00.dat" using 1:(2.0*\$5) with lines ls 24, \
    "$prb_root/Cs/prb_curves_U6.00.dat" using 1:(3.0*\$2) with lines ls 31, \
    "$prb_root/Cs/prb_curves_U6.00.dat" using 1:(3.0*\$3) with lines ls 32, \
    "$prb_root/Cs/prb_curves_U6.00.dat" using 1:(3.0*\$4) with lines ls 33, \
    "$prb_root/Cs/prb_curves_U6.00.dat" using 1:(3.0*\$5) with lines ls 34
unset multiplot
EOF
}

run_prb_variant() {
    local variant_name="$1"
    local projector_mode="$2"
    local variant_root="$OUTPUT_ROOT/$variant_name"
    local log_file="$LOG_DIR/${variant_name}.log"
    local projector_file="$variant_root/projector.dat"
    local site_dir site zeta U ratio heff_txt jmu_txt line
    : > "$log_file"
    mkdir -p "$variant_root"

    if [ "$projector_mode" = "manual" ]; then
        write_prb_projector "$projector_file"
    else
        projector_file="$variant_root/projector_oh_j52.npz"
        write_prb_oh_j52_projector "$projector_file" "$variant_root/cef_states.log"
    fi

    for site in K Rb Cs; do
        if [ "$SMOKE" -eq 1 ] && [ "$site" != "K" ]; then
            continue
        fi
        site_dir="$variant_root/$site"
        mkdir -p "$site_dir"
        (
            cd "$site_dir"
            run_repo_python "$TOOLS_DIR/slater_koster_pf.py" literature --output "prb_t_mu.dat" > "slater_koster.log" 2>&1
        )
        if [ "$site" = "K" ]; then
            zeta="120.0"
        else
            zeta="110.0"
        fi

        for U in 2.0 3.0 4.0 6.0; do
            if [ "$SMOKE" -eq 1 ] && [ "$U" != "4.0" ]; then
                continue
            fi
            printf "#ratio J K Gamma GammaPrime symmetry_leakage\n" > "$site_dir/prb_curves_U$(printf "%.2f" "$U").dat"
            for ratio in $(seq -f "%.2f" 0.00 0.01 0.20); do
                if [ "$SMOKE" -eq 1 ] && [ "$ratio" != "0.10" ]; then
                    continue
                fi
                write_run_input "$site_dir/run_input.toml" "${site}_U${U}_r${ratio}" "${site}_U${U}_r${ratio}" 1 "12.980 8.163 5.878" "$U" "$ratio" "$zeta" \
                    "$site_dir/prb_t_mu_${site}.dat" "$projector_file"
                (
                    cd "$site_dir"
                    run_repo_python "$REPO_ROOT/fexchange/cli.py" run "run_input.toml" >> "cli.log" 2>&1
                )
                heff_txt="$site_dir/heff.txt"
                jmu_txt="$site_dir/jmu.txt"
                dump_heff_to_jmu "$site_dir/$L4_REL" "$heff_txt" "$jmu_txt"
                line="$(awk -v ratio="$ratio" -f "$PRB_CHANNELS_AWK" "$jmu_txt")"
                printf "%s\n" "$line" >> "$site_dir/prb_curves_U$(printf "%.2f" "$U").dat"
            done
        done
    done

    if [ "$SMOKE" -eq 0 ]; then
        plot_prb "$variant_root" "$variant_root/prb_particle_plot.png" >> "$log_file" 2>&1
    fi
}

run_prb() {
    run_prb_variant "prb" "manual"
    run_prb_variant "prb_oh_j52" "oh_j52"
}

if [ "$TASK" = "all" ] || [ "$TASK" = "ybox" ]; then
    run_ybox
fi

if [ "$TASK" = "all" ] || [ "$TASK" = "prb" ]; then
    run_prb
fi

echo "Validation artifacts written to: $OUTPUT_ROOT"
if [ -f "$OUTPUT_ROOT/prb/prb_particle_plot.png" ]; then
    echo "PRB plot: $OUTPUT_ROOT/prb/prb_particle_plot.png"
fi
if [ -f "$OUTPUT_ROOT/prb_oh_j52/prb_particle_plot.png" ]; then
    echo "PRB Oh J=5/2 plot: $OUTPUT_ROOT/prb_oh_j52/prb_particle_plot.png"
fi
