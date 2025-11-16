
clang rabbit_construction_running_rime.cpp
for n in 1000 2000 3000 4000 5000 ; do ( echo ; echo $n ; time ./a.out $n ) >>file.txt 2>&1 ; done

