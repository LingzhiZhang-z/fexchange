set macros

if (!exists("curve_root")) curve_root = "outputs/nature_fig3_particle/fig3"
if (!exists("reference_root")) reference_root = "data/test/nature_fig3_particle/reference"
if (!exists("output_file")) output_file = "outputs/nature_fig3_particle/fig3/nature_fig3_abef.png"
if (!exists("term_spec")) term_spec = "pngcairo size 1400,980 enhanced font 'Helvetica,18'"

set terminal @term_spec
set output output_file
set datafile commentschars "#"
set multiplot layout 2,2 rowsfirst
set key outside top center horizontal width 1
set grid lc rgb "#d9d9d9" lw 0.8
set border lw 1.0
set xrange [0.1:1.0]
set xtics 0.1,0.1,1.0
set mxtics 2
set style line 1 lc rgb "#111111" lw 2
set style line 2 lc rgb "#cf3c2f" lw 2
set style line 3 lc rgb "#f08a24" lw 2
set style line 4 lc rgb "#5566cc" lw 2
set style line 11 lc rgb "#111111" pt 6 ps 0.7 lw 1.2
set style line 12 lc rgb "#cf3c2f" pt 6 ps 0.7 lw 1.2
set style line 13 lc rgb "#f08a24" pt 6 ps 0.7 lw 1.2
set style line 14 lc rgb "#5566cc" pt 6 ps 0.7 lw 1.2

set ylabel "meV"
set xlabel ""
set title "(a) 4f^1 Gamma7" noenhanced
plot \
    curve_root."/f1_ce_g7/curve.tsv" using 1:2 with lines ls 1 title "J", \
    reference_root."/f1_ce_g7.tsv" using 1:2 with points ls 11 notitle, \
    curve_root."/f1_ce_g7/curve.tsv" using 1:3 with lines ls 2 title "K", \
    reference_root."/f1_ce_g7.tsv" using 1:3 with points ls 12 notitle, \
    curve_root."/f1_ce_g7/curve.tsv" using 1:4 with lines ls 3 title "Gamma", \
    reference_root."/f1_ce_g7.tsv" using 1:4 with points ls 13 notitle, \
    curve_root."/f1_ce_g7/curve.tsv" using 1:5 with lines ls 4 title "GammaPrime", \
    reference_root."/f1_ce_g7.tsv" using 1:5 with points ls 14 notitle

set ylabel ""
set title "(b) 4f^3 Gamma6" noenhanced
plot \
    curve_root."/f3_nd_g6/curve.tsv" using 1:2 with lines ls 1 notitle, \
    reference_root."/f3_nd_g6.tsv" using 1:2 with points ls 11 notitle, \
    curve_root."/f3_nd_g6/curve.tsv" using 1:3 with lines ls 2 notitle, \
    reference_root."/f3_nd_g6.tsv" using 1:3 with points ls 12 notitle, \
    curve_root."/f3_nd_g6/curve.tsv" using 1:4 with lines ls 3 notitle, \
    reference_root."/f3_nd_g6.tsv" using 1:4 with points ls 13 notitle, \
    curve_root."/f3_nd_g6/curve.tsv" using 1:5 with lines ls 4 notitle, \
    reference_root."/f3_nd_g6.tsv" using 1:5 with points ls 14 notitle

set ylabel "meV"
set xlabel "|pfpi/pfsigma|"
set title "(e) 4f^11 Gamma7" noenhanced
plot \
    curve_root."/f11_er_g7/curve.tsv" using 1:2 with lines ls 1 notitle, \
    reference_root."/f11_er_g7.tsv" using 1:2 with points ls 11 notitle, \
    curve_root."/f11_er_g7/curve.tsv" using 1:3 with lines ls 2 notitle, \
    reference_root."/f11_er_g7.tsv" using 1:3 with points ls 12 notitle, \
    curve_root."/f11_er_g7/curve.tsv" using 1:4 with lines ls 3 notitle, \
    reference_root."/f11_er_g7.tsv" using 1:4 with points ls 13 notitle, \
    curve_root."/f11_er_g7/curve.tsv" using 1:5 with lines ls 4 notitle, \
    reference_root."/f11_er_g7.tsv" using 1:5 with points ls 14 notitle

set ylabel ""
set title "(f) 4f^13 Gamma6" noenhanced
plot \
    curve_root."/f13_yb_g6/curve.tsv" using 1:2 with lines ls 1 notitle, \
    reference_root."/f13_yb_g6.tsv" using 1:2 with points ls 11 notitle, \
    curve_root."/f13_yb_g6/curve.tsv" using 1:3 with lines ls 2 notitle, \
    reference_root."/f13_yb_g6.tsv" using 1:3 with points ls 12 notitle, \
    curve_root."/f13_yb_g6/curve.tsv" using 1:4 with lines ls 3 notitle, \
    reference_root."/f13_yb_g6.tsv" using 1:4 with points ls 13 notitle, \
    curve_root."/f13_yb_g6/curve.tsv" using 1:5 with lines ls 4 notitle, \
    reference_root."/f13_yb_g6.tsv" using 1:5 with points ls 14 notitle

unset multiplot
set output
