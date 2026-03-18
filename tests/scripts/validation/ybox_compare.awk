function shq(s,    out, i, c) {
    out = "'"
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "'") {
            out = out "'\\''"
        } else {
            out = out c
        }
    }
    return out "'"
}

function absval(x) {
    return x < 0 ? -x : x
}

function load_reference(    line, n, vals) {
    while ((getline line < ref_file) > 0) {
        if (line ~ /^#/ || line ~ /^[ \t]*$/) {
            continue
        }
        n = split(line, vals, /[ \t]+/)
        if (absval(vals[3] - ratio) < 1.0e-12) {
            ref[0,0] = vals[4]
            ref[0,1] = vals[5]
            ref[0,2] = vals[6]
            ref[1,0] = vals[7]
            ref[1,1] = vals[8]
            ref[1,2] = vals[9]
            ref[2,0] = vals[10]
            ref[2,1] = vals[11]
            ref[2,2] = vals[12]
            close(ref_file)
            return
        }
    }
    close(ref_file)
    print "reference row not found for ratio=" ratio > "/dev/stderr"
    exit 1
}

function init_signs() {
    label[0] = "identity"
    sx[0] = 1; sy[0] = 1; sz[0] = 1
    label[1] = "flip_xz"
    sx[1] = -1; sy[1] = 1; sz[1] = -1
    label[2] = "flip_yz"
    sx[2] = 1; sy[2] = -1; sz[2] = -1
    label[3] = "flip_xy"
    sx[3] = -1; sy[3] = -1; sz[3] = 1
}

function sign_for(label_name, axis) {
    if (label_name == "identity") {
        return 1
    }
    if (label_name == "flip_xz") {
        return axis == 1 ? 1 : -1
    }
    if (label_name == "flip_yz") {
        return axis == 0 ? 1 : -1
    }
    if (label_name == "flip_xy") {
        return axis == 2 ? 1 : -1
    }
    print "unknown gauge " label_name > "/dev/stderr"
    exit 1
}

{
    for (col = 1; col <= NF; col++) {
        J[row_idx, col - 1] = $col + 0.0
    }
    row_idx++
}

BEGIN {
    row_idx = 0
}

END {
    load_reference()
    init_signs()

    if (mode == "best") {
        best_label = "identity"
        best_err = 1.0e99
        for (g = 0; g < 4; g++) {
            max_err = 0.0
            for (i = 0; i < 3; i++) {
                si = (i == 0 ? sx[g] : (i == 1 ? sy[g] : sz[g]))
                for (j = 0; j < 3; j++) {
                    sj = (j == 0 ? sx[g] : (j == 1 ? sy[g] : sz[g]))
                    value = si * sj * J[i,j]
                    diff = absval(value - ref[i,j])
                    if (diff > max_err) {
                        max_err = diff
                    }
                }
            }
            if (max_err < best_err) {
                best_err = max_err
                best_label = label[g]
            }
        }
        printf("GAUGE_LABEL=%s\n", shq(best_label))
        printf("GAUGE_ERR=%.12e\n", best_err)
        exit 0
    }

    raw_err = 0.0
    aligned_err = 0.0
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            raw_diff = absval(J[i,j] - ref[i,j])
            if (raw_diff > raw_err) {
                raw_err = raw_diff
            }
            value = sign_for(gauge, i) * sign_for(gauge, j) * J[i,j]
            aligned[i,j] = value
            aligned_diff = absval(value - ref[i,j])
            if (aligned_diff > aligned_err) {
                aligned_err = aligned_diff
            }
        }
    }

    Jh = U * ratio
    raw_line = sprintf("%.6f %.6f %.6f", U, Jh, ratio)
    aligned_line = sprintf("%.6f %.6f %.6f", U, Jh, ratio)
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            raw_line = raw_line sprintf(" %.6f", J[i,j])
            aligned_line = aligned_line sprintf(" %.6f", aligned[i,j])
        }
    }
    raw_line = raw_line sprintf(" %.6f", raw_err)
    aligned_line = aligned_line sprintf(" %.6f", aligned_err)

    printf("RAW_ERR=%.12e\n", raw_err)
    printf("ALIGNED_ERR=%.12e\n", aligned_err)
    printf("RAW_ROW=%s\n", shq(raw_line))
    printf("ALIGNED_ROW=%s\n", shq(aligned_line))
}
