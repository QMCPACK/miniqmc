set term png size 1200,700 enhanced font "Terminal,12"
set output "plot.png"
set grid

set datafile separator ";"

set auto x

set style data histogram
set style fill solid border -1
set boxwidth 0.9

set xtic rotate by -44 scale 0

set multiplot layout 1, 1 rowsfirst

set key right
set yrange [0:100]
set ylabel "Time (ms)"
set xlabel "Running on 'Nb' mpi process and 'Nb' omp threads "
set title "Scalability analysis of the miniQMC application "
plot "scalabilite_bench.dat" using (1/$1)*100 :xticlabels(stringcolumn(1)) t "mpi+threads"

