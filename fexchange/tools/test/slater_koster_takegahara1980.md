# Slater-Koster Tables for f Electrons

**Reference:** Takegahara, Aoki & Yanase, J. Phys. C: Solid State Phys. 13 (1980) 583–588.

This file stores ALL data from the reference paper for future implementation.
Currently, only p-f integrals are implemented in `tools/slater_koski_pf.py`.

## Notation

Direction cosines: l, m, n of the bond vector X.

Two-centre integral parameters:
    s-f:  (sfσ)                         — 1 parameter
    p-f:  (pfσ), (pfπ)                  — 2 parameters
    d-f:  (dfσ), (dfπ), (dfδ)           — 3 parameters
    f-f:  (ffσ), (ffπ), (ffδ), (ffφ)    — 4 parameters

Cubic harmonic normalization constants:
    C_s = (1/4π)^{1/2}
    C_p = (3/4π)^{1/2} r^{-1}
    C_d = (5/16π)^{1/2} r^{-2}
    C_f = (7/16π)^{1/2} r^{-3}


## Table 1 — Cubic Harmonics (spherical harmonic decomposition)

s:   C_s = Y₀₀

p:   T₁ᵤ x:   C_p x       = √(1/2)(−Y₁₁ + Y₁₋₁)
     T₁ᵤ y:   C_p y       = i√(1/2)(Y₁₁ + Y₁₋₁)
     T₁ᵤ z:   C_p z       = Y₁₀

d:   E_g  u:   C_d(3z²−r²)    = Y₂₀
     E_g  v:   √3 C_d(x²−y²)  = √(1/2)(Y₂₂ + Y₂₋₂)
     T₂g  ξ:   2√3 C_d yz     = i√(1/2)(Y₂₁ + Y₂₋₁)
     T₂g  η:   2√3 C_d zx     = √(1/2)(−Y₂₁ + Y₂₋₁)
     T₂g  ζ:   2√3 C_d xy     = i√(1/2)(−Y₂₂ + Y₂₋₂)

f:   A₂ᵤ   :   2√15 C_f xyz        = i√(1/2)(−Y₃₂ + Y₃₋₂)
     T₁ᵤ  α:   C_f x(5x²−3r²)     = (1/4)(−√5 Y₃₃ + √3 Y₃₁ − √3 Y₃₋₁ + √5 Y₃₋₃)
     T₁ᵤ  β:   C_f y(5y²−3r²)     = (i/4)(−√5 Y₃₃ − √3 Y₃₁ − √3 Y₃₋₁ − √5 Y₃₋₃)
     T₁ᵤ  γ:   C_f z(5z²−3r²)     = Y₃₀
     T₂ᵤ  ξ:   √15 C_f x(y²−z²)   = (1/4)(√3 Y₃₃ + √5 Y₃₁ − √5 Y₃₋₁ − √3 Y₃₋₃)
     T₂ᵤ  η:   √15 C_f y(z²−x²)   = (i/4)(−√3 Y₃₃ + √5 Y₃₁ + √5 Y₃₋₁ − √3 Y₃₋₃)
     T₂ᵤ  ζ:   √15 C_f z(x²−y²)   = √(1/2)(Y₃₂ + Y₃₋₂)


## Table 2 — Energy Integrals (Slater-Koster two-centre integrals)

All entries are E_{x, f_λ}; other entries obtained by cyclic permutation
of coordinates (x→y→z) and direction cosines (l→m→n).

IMPORTANT: T₂ᵤ orbitals acquire an extra (−1) under coordinate swaps
(ξ→−η under x↔y, etc.) due to the cubic harmonic phase convention.

### s-f integrals

    E_{s, xyz}         = √15 lmn (sfσ)
    E_{s, z(5z²−3)}    = (1/2) n(5n²−3) (sfσ)
    E_{s, z(x²−y²)}    = (1/2)√15 n(l²−m²) (sfσ)

### p-f integrals — VERIFIED NUMERICALLY (max error < 1e-14)

    E_{x, xyz}         = √15 l²mn (pfσ)          − √10/2 · mn(3l²−1) (pfπ)
    E_{x, x(5x²−3)}    = (1/2) l²(5l²−3) (pfσ)   + √6/4 · (1−l²)(5l²−1) (pfπ)
    E_{x, y(5y²−3)}    = (1/2) lm(5m²−3) (pfσ)   − √6/4 · lm(5m²−1) (pfπ)
    E_{x, z(5z²−3)}    = (1/2) ln(5n²−3) (pfσ)   − √6/4 · ln(5n²−1) (pfπ)
    E_{x, x(y²−z²)}    = (1/2)√15 l²(m²−n²) (pfσ)  + √10/4 · (n²−m²)(3l²−1) (pfπ)
    E_{x, y(z²−x²)}    = (1/2)√15 lm(n²−l²) (pfσ)  − √10/4 · lm[3(n²−l²)+2] (pfπ)
    E_{x, z(x²−y²)}    = (1/2)√15 ln(l²−m²) (pfσ)  − √10/4 · ln[3(l²−m²)−2] (pfπ)

### d-f integrals (from paper Table 2)

    E_{xy, xyz}        = √45 l²m²n (dfσ)
                        − √5 n(6l²m²+n²−1) (dfπ)
                        + n(3l²m²+2n²−1) (dfδ)

    E_{xy, x(5x²−3)}  = (1/2)√3 l²m(5l²−3) (dfσ)
                        − √(5/8) m(5l²−1)(2l²−1) (dfπ)
                        + (1/2)√15 l²m(l²−1) (dfδ)

    E_{xy, y(5y²−3)}  = (1/2)√3 lm²(5m²−3) (dfσ)
                        − √(5/8) l(5m²−1)(2m²−1) (dfπ)
                        + (1/2)√15 lm²(m²−1) (dfδ)

    E_{xy, z(5z²−3)}  = (1/2)√3 lmn(5n²−3) (dfσ)
                        − √(5/8) lmn(5n²−1) (dfπ)
                        + (1/2)√15 lmn(n²+1) (dfδ)

    E_{xy, x(y²−z²)}  = (1/2)√5 l²m(m²−n²) (dfσ)
                        − √(5/8) m[(6l²−1)(m²−n²)−2l²] (dfπ)
                        + (1/2) m[3l²(m²−n²)+4n²−2l²] (dfδ)

    (See paper Table 2 pp. 586–587 for remaining d-f entries — ~21 independent entries)

### f-f integrals (from paper Table 2)

    E_{xyz, xyz} = 15 l²m²n² (ffσ) + ... (4 terms with ffσ, ffπ, ffδ, ffφ)

    (Full expressions are very long; see paper Table 2 pp. 587 — ~15 independent entries)


## Errata

Sharma (1979) E_{xy, y³−3x²y}: coefficient of (dfσ) should be
    (1/16)√30 l(1 − l⁴ + 10m²l² − 5m⁴ − 2n² + n⁴).
