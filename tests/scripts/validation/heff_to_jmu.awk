function sigma_re(label, row, col) {
    if (label == "x") {
        return ((row == 0 && col == 1) || (row == 1 && col == 0)) ? 1.0 : 0.0
    }
    if (label == "y") {
        return 0.0
    }
    if (label == "z") {
        if (row == 0 && col == 0) {
            return 1.0
        }
        if (row == 1 && col == 1) {
            return -1.0
        }
    }
    return 0.0
}

function sigma_im(label, row, col) {
    if (label == "y") {
        if (row == 0 && col == 1) {
            return -1.0
        }
        if (row == 1 && col == 0) {
            return 1.0
        }
    }
    return 0.0
}

/^#/ {
    next
}

{
    Hre[$1 "," $2] = $3
    Him[$1 "," $2] = $4
}

END {
    labels[0] = "x"
    labels[1] = "y"
    labels[2] = "z"
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
            tr_re = 0.0
            for (i = 0; i < 4; i++) {
                i1 = int(i / 2)
                i2 = i % 2
                for (k = 0; k < 4; k++) {
                    k1 = int(k / 2)
                    k2 = k % 2
                    s1r = sigma_re(labels[a], i1, k1)
                    s1i = sigma_im(labels[a], i1, k1)
                    s2r = sigma_re(labels[b], i2, k2)
                    s2i = sigma_im(labels[b], i2, k2)
                    br = s1r * s2r - s1i * s2i
                    bi = s1r * s2i + s1i * s2r
                    hr = Hre[k "," i]
                    hi = Him[k "," i]
                    tr_re += br * hr - bi * hi
                }
            }
            J[a "," b] = tr_re
        }
    }

    for (i = 0; i < 3; i++) {
        printf("%.12f %.12f %.12f\n", J[i ",0"], J[i ",1"], J[i ",2"])
    }
}
