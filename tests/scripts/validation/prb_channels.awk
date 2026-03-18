function absval(x) {
    return x < 0 ? -x : x
}

function signval(x) {
    return x < 0 ? -1.0 : 1.0
}

function cross(ax, ay, az, bx, by, bz, out_key) {
    vec[out_key ",0"] = ay * bz - az * by
    vec[out_key ",1"] = az * bx - ax * bz
    vec[out_key ",2"] = ax * by - ay * bx
}

function norm3(key) {
    return sqrt(vec[key ",0"] * vec[key ",0"] + vec[key ",1"] * vec[key ",1"] + vec[key ",2"] * vec[key ",2"])
}

function normalize3(key,    n, i) {
    n = norm3(key)
    if (n < 1.0e-14) {
        return 0
    }
    for (i = 0; i < 3; i++) {
        vec[key "," i] /= n
    }
    return 1
}

function det3(    value) {
    value = R[0,0] * (R[1,1] * R[2,2] - R[1,2] * R[2,1]) \
          - R[0,1] * (R[1,0] * R[2,2] - R[1,2] * R[2,0]) \
          + R[0,2] * (R[1,0] * R[2,1] - R[1,1] * R[2,0])
    return value
}

function jacobi(    i, j, p, q, max_off, theta, t, c, s, app, aqq, apq, apj, aqj, vpj, vqj, iter) {
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            V[i,j] = (i == j ? 1.0 : 0.0)
        }
    }

    for (iter = 0; iter < 50; iter++) {
        max_off = 0.0
        p = 0
        q = 1
        for (i = 0; i < 3; i++) {
            for (j = i + 1; j < 3; j++) {
                if (absval(A[i,j]) > max_off) {
                    max_off = absval(A[i,j])
                    p = i
                    q = j
                }
            }
        }
        if (max_off < 1.0e-12) {
            break
        }

        theta = (A[q,q] - A[p,p]) / (2.0 * A[p,q])
        if (theta == 0.0) {
            t = 1.0
        } else {
            t = signval(theta) / (absval(theta) + sqrt(theta * theta + 1.0))
        }
        c = 1.0 / sqrt(t * t + 1.0)
        s = t * c

        app = A[p,p]
        aqq = A[q,q]
        apq = A[p,q]
        A[p,p] = app - t * apq
        A[q,q] = aqq + t * apq
        A[p,q] = 0.0
        A[q,p] = 0.0

        for (j = 0; j < 3; j++) {
            if (j != p && j != q) {
                apj = A[p,j]
                aqj = A[q,j]
                A[p,j] = c * apj - s * aqj
                A[j,p] = A[p,j]
                A[q,j] = s * apj + c * aqj
                A[j,q] = A[q,j]
            }
        }

        for (j = 0; j < 3; j++) {
            vpj = V[j,p]
            vqj = V[j,q]
            V[j,p] = c * vpj - s * vqj
            V[j,q] = s * vpj + c * vqj
        }
    }
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
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Jsym[i,j] = 0.5 * (J[i,j] + J[j,i])
            A[i,j] = Jsym[i,j]
        }
    }

    jacobi()
    axis_idx = 0
    best_abs_z = absval(V[2,0])
    for (i = 1; i < 3; i++) {
        if (absval(V[2,i]) < best_abs_z) {
            axis_idx = i
            best_abs_z = absval(V[2,i])
        }
    }

    for (i = 0; i < 3; i++) {
        vec["u," i] = V[i,axis_idx]
    }
    if (vec["u,0"] - vec["u,1"] < 0.0) {
        for (i = 0; i < 3; i++) {
            vec["u," i] = -vec["u," i]
        }
    }

    dot = vec["u,2"]
    vec["z,0"] = -vec["u,0"] * dot
    vec["z,1"] = -vec["u,1"] * dot
    vec["z,2"] = 1.0 - vec["u,2"] * dot
    if (!normalize3("z")) {
        dot = vec["u,0"]
        vec["z,0"] = 1.0 - vec["u,0"] * dot
        vec["z,1"] = -vec["u,1"] * dot
        vec["z,2"] = -vec["u,2"] * dot
        normalize3("z")
    }

    cross(vec["z,0"], vec["z,1"], vec["z,2"], vec["u,0"], vec["u,1"], vec["u,2"], "v")
    normalize3("v")

    sq2 = sqrt(2.0)
    for (i = 0; i < 3; i++) {
        vec["x," i] = (vec["u," i] + vec["v," i]) / sq2
        vec["y," i] = (vec["v," i] - vec["u," i]) / sq2
        vec["z2," i] = vec["z," i]
    }

    for (i = 0; i < 3; i++) {
        R[i,0] = vec["x," i]
        R[i,1] = vec["y," i]
        R[i,2] = vec["z2," i]
    }

    if (det3() < 0.0) {
        for (i = 0; i < 3; i++) {
            vec["z," i] = -vec["z," i]
        }
        cross(vec["z,0"], vec["z,1"], vec["z,2"], vec["u,0"], vec["u,1"], vec["u,2"], "v")
        normalize3("v")
        for (i = 0; i < 3; i++) {
            vec["x," i] = (vec["u," i] + vec["v," i]) / sq2
            vec["y," i] = (vec["v," i] - vec["u," i]) / sq2
            R[i,0] = vec["x," i]
            R[i,1] = vec["y," i]
            R[i,2] = vec["z," i]
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Jcan[i,j] = 0.0
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    Jcan[i,j] += R[a,i] * Jsym[a,b] * R[b,j]
                }
            }
        }
    }

    Jiso = 0.5 * (Jcan[0,0] + Jcan[1,1])
    K = Jcan[2,2] - Jiso
    Gamma = 0.5 * (Jcan[0,1] + Jcan[1,0])
    GammaP = 0.25 * (Jcan[0,2] + Jcan[2,0] + Jcan[1,2] + Jcan[2,1])
    leakage = absval(Jcan[0,0] - Jcan[1,1])
    if (absval(Jcan[0,1] - Jcan[1,0]) > leakage) {
        leakage = absval(Jcan[0,1] - Jcan[1,0])
    }
    if (absval(Jcan[0,2] - Jcan[1,2]) > leakage) {
        leakage = absval(Jcan[0,2] - Jcan[1,2])
    }
    if (absval(Jcan[2,0] - Jcan[2,1]) > leakage) {
        leakage = absval(Jcan[2,0] - Jcan[2,1])
    }

    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", ratio, Jiso, K, Gamma, GammaP, leakage)
}
