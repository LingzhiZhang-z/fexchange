#!/bin/bash
# End-to-end CLI test: w90_extract.py -> disk -> w90_decompose_h_local.py


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$TOOLS_DIR")/data"
WORK_DIR=$(mktemp -d)

trap "rm -rf $WORK_DIR" EXIT

PASS=0
FAIL=0

check() {
    local label="$1" ok="$2"
    if [ "$ok" = "true" ]; then
        echo "  PASS $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $label"
        FAIL=$((FAIL + 1))
    fi
}

run_case() {
    local mat="$1" bond="$2" mode="$3"
    local ai="$4" ci="$5" aj="$6" cj="$7"
    local l1a="$8" l1c="$9" l2a="${10}" l2c="${11}"
    local ref_soc="${12}"

    local label="${mat}_${mode}_${bond}_bond1"
    local prefix="$WORK_DIR/$label"

    if [ "$mode" = "soc" ]; then
        local hr="$DATA_DIR/$mat/SOC/wannier_hr.dat"
        local ndims="[14,14,14,14,14,14,6,6,6,6,6,6,6,6,6,6,6,6]"
        local spinor_flag="true"
    else
        local hr="$DATA_DIR/$mat/NSOC/wannier_hr.dat"
        local ndims="[7,7,7,7,7,7,3,3,3,3,3,3,3,3,3,3,3,3]"
        local spinor_flag="false"
    fi

    # Step 1: generate toml
    cat > "${prefix}.toml" <<EOF
hr_path = "$hr"
spinor = $spinor_flag
energy_unit = "eV"
n_orb_per_atom = $ndims
output = "$prefix"

[f_site_i]
atom = $ai
cell = $ci

[f_site_j]
atom = $aj
cell = $cj

[[ligands]]
atom = $l1a
cell = $l1c

[[ligands]]
atom = $l2a
cell = $l2c
EOF

    # Step 2: run w90_extract CLI
    local extract_err
    extract_err=$(python "$TOOLS_DIR/w90_extract.py" "${prefix}.toml" 2>&1)
    if [ $? -ne 0 ]; then
        echo "  FAIL $label: w90_extract error:"
        echo "$extract_err" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return
    fi

    # Step 3: check output files exist with correct shape
    local t_mu_file="${prefix}_t_mu.dat"
    local h_local_file="${prefix}_h_local.dat"
    local t_ok="false"
    if [ -f "$t_mu_file" ]; then
        local rows
        rows=$(wc -l < "$t_mu_file" | tr -d ' ')
        [ "$rows" = "14" ] && t_ok="true"
    fi
    check "$label: t_mu 14x14" "$t_ok"

    # Step 4: run w90_decompose_h_local CLI
    local output
    output=$(python "$TOOLS_DIR/w90_decompose_h_local.py" --h_local_dat "$h_local_file" --rtol 1e-2 2>&1)
    if [ $? -ne 0 ]; then
        echo "  FAIL $label: decompose error:"
        echo "$output" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return
    fi

    # extract zeta value
    local zeta
    zeta=$(echo "$output" | sed -n 's/.*zeta = \([0-9.]*\).*/\1/p')

    if [ "$mode" = "soc" ]; then
        # check h_cef file
        local h_cef_file="${prefix}_h_cef.dat"
        local cef_ok="false"
        if [ -f "$h_cef_file" ]; then
            local cef_rows
            cef_rows=$(wc -l < "$h_cef_file" | tr -d ' ')
            [ "$cef_rows" = "7" ] && cef_ok="true"
        fi
        check "$label: h_cef 7x7" "$cef_ok"

        # check zeta vs reference
        local zeta_ok
        zeta_ok=$(python -c "
z, r = float('$zeta'), float('$ref_soc')
print('true' if abs(z - r) / r < 0.01 else 'false')
")
        check "$label: zeta=${zeta} vs ref=${ref_soc}" "$zeta_ok"
    else
        local zeta_ok="false"
        [ "$zeta" = "0.000000" ] && zeta_ok="true"
        check "$label: zeta=0 (nsoc)" "$zeta_ok"
    fi
}

echo "============================================================"
echo "CLI end-to-end test"
echo "============================================================"

# YbOCl
run_case YbOCl 1st soc  0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"    393.2686
run_case YbOCl 1st nsoc 0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"    393.2686
run_case YbOCl 2nd soc  0 "[0,0,0]" 0 "[1,0,0]" 6 "[0,0,0]" 12 "[0,-1,0]"  393.2686
run_case YbOCl 2nd nsoc 0 "[0,0,0]" 0 "[1,0,0]" 6 "[0,0,0]" 12 "[0,-1,0]"  393.2686
# YbOBr
run_case YbOBr 1st soc  0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"    391.7807
run_case YbOBr 1st nsoc 0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"    391.7807
run_case YbOBr 2nd soc  0 "[0,0,0]" 0 "[1,0,0]" 6 "[0,0,0]" 12 "[0,-1,0]"  391.7807
run_case YbOBr 2nd nsoc 0 "[0,0,0]" 0 "[1,0,0]" 6 "[0,0,0]" 12 "[0,-1,0]"  391.7807
# YbOF
run_case YbOF 1st soc  0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"     393.8325
run_case YbOF 1st nsoc 0 "[0,0,0]" 3 "[0,0,0]" 6 "[0,0,0]" 7 "[0,0,0]"     393.8325
run_case YbOF 2nd soc  0 "[0,0,0]" 0 "[-1,0,0]" 6 "[0,0,0]" 12 "[0,0,0]"   393.8325
run_case YbOF 2nd nsoc 0 "[0,0,0]" 0 "[-1,0,0]" 6 "[0,0,0]" 12 "[0,0,0]"   393.8325

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
