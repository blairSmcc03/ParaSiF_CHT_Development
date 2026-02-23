#set terminal postfile       (These commented lines would be used to )
#set output  "d1_plot.ps"    (generate a postscript file.            )
set title "Temperature along y=0.5m" font "Arial, 36"
set xlabel "x(m)" font "Arial, 28"
set ylabel "Temp (K)" font "Arial,28"
set yrange[273.15:274.15]
set ytics 0.2
set xtics 0.2
set xtics font "Arial,20"
set ytics font "Arial,20"
set key right bottom
set key font "Arial,28"


plot "heatSolverPy-OpenFoamT=70.xy" with lines dt 1 lw 6 lc rgb "red" title "OpenFOAM-heatSolverPy"

replot "../../../../FSI_case/caseA/validation/chtMultiRegionFoamT\=74.64.xy" with lines dt 8 lw 6 lc rgb "blue" title "chtMultiRegionFoam"

replot "refCaseA.xy" u ($1):($2+273.15) with points pointtype 8 ps 6 lc rgb "black" title "Reference"

pause -1 "Hit any key to continue"
