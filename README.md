# fast-kappa-map

"python fastermap.py" runs on a single core in around 8 minutes and produces the file "mpfast.npy"

"mpiexec -n x python mpimap.py" runs on x cores in ..?.. minutes ( < 8) and produces the file "mp_mpi.npy"\
	mpimap might not print anything for a while (not sure why but running "time mpiexec..." shows that there's a lot of time spent on sys commands)

mpfast.npy and mp__mpi.py are the mpoints after the least squares solver

