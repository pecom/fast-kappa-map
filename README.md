# fast-kappa-map

"python fastermap.py" runs on a single core in around 8 minutes and produces the file "mpfast.npy"

"mpiexec -n x python mpimap.py" runs on x cores in ..?.. minutes ( < 8) and produces the file "mp_mpi.npy"\
	mpimap might not print anything for a while (not sure why but running "time mpiexec..." shows that there's a lot of time spent on sys commands)

mpfast.npy and mp__mpi.py are the mpoints after the least squares solver

Following Sam's naming scheme, cl_auto/cross_Nside_mid_sigfactor (using midpoint) or cl_auto/cross_Nside_srad_sigfactor (using search radius)

Use config.json to set up automated tests. Keep them all the same length aside from "range"
  - srad = Array of float (1 to 15 which corresponds to .1 degrees to 1.5 degrees), search radius (if midpoint is False)
  - midpoint = Array of boolean, if True, will override srad (0 for False, 1 for True)
  - Nside = Array of int (64 or 128)
  - sigfactor = Array of float (1 to ?), signal damping factor.
  - range = True or False (0 for False, 1 for True). If true, will do n^4 runs (all possible combinations of previous factors). If false, will do n runs

