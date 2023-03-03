set term png size 1700,700 enhanced font "Terminal,12"
set output "plot.png"
#set grid

set datafile separator ";"
set auto x

set style data histogram
set style fill solid border -1
set boxwidth 0.9

set xtic rotate by -44 scale 0
set logscale y
set yrange [0.01:10000]
set multiplot layout 1, 1 rowsfirst

set key left
set ylabel "Time (s)"
set xlabel "Running on 'Nb' mpi process and 'Nb' omp threads"
set title "Scalability analysis of the miniQMC application with differents problems size"
plot "scalabilite_bench211.dat" using 2:xticlabels(stringcolumn(1)) t "miniQMC+211",\
	 "scalabilite_bench221.dat" using 2:xticlabels(stringcolumn(1)) t "miniQMC+221",\
	 "scalabilite_bench222.dat" using 2:xticlabels(stringcolumn(1)) t "miniQMC+222",\
	 "scalabilite_bench422.dat" using 2:xticlabels(stringcolumn(1)) t "miniQMC+422",\
	 

