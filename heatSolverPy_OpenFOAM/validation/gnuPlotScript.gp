#set terminal postfile       (These commented lines would be used to )
#set output  "d1_plot.ps"    (generate a postscript file.            )
set title "Temperature at y=0.5m, t=70s"
set xlabel "x(m)"
set ylabel "Temp (C)"
set yrange[273:274.5]
set ytics 0.2
set xtics 0.2
set key left top

plot "refCaseA.xy" u ($1):($2+273.15) with points pointtype 8 ps 2 lc rgb "red" title "Reference"


replot "ofTOPythondata.xy" with lines title "OpenFoam Python Coupling"
replot "chtMultiRegionFoamData.xy" with lines title "chtMultiRegionFoam"

pause -1 "Hit any key to continue"
