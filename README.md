# fast-kappa-map

Currently doesn't take any arguments and reads from DRQ_mocks.fits (line 15) and kappalist (line 39) for now.\
Will add functionality to take file names for Quasar class and Data class soon.\

"python fastermap.py" runs on a single core in around 8 minutes and produces the file "mpfast.npy"\

"mpiexec -n x python mpimap.py" runs on x cores in ..?.. minutes ( < 8) and produces the file "mp_mpi.npy"\
	mpimap might not print anything for a while (not sure why but running "time mpiexec..." shows that there's a lot of time spent on sys commands) \
