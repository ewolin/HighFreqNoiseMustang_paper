#!/usr/bin/env python
# Interpolate piecewise linear models at same spacing as Peterson model files.
# Note that input files go from long to short period
# so we need to reverse the order of input arrays to np.interp
import sys
import numpy as np

peterson_model = sys.argv[1]
piecewise_model = sys.argv[2]
stitch_model = sys.argv[3]

# Read Peterson noise model
t, db = np.loadtxt(peterson_model, unpack=True)
tstep = (np.log10(t.max()) - np.log10(t.min()))/len(t)
# tstep should be ~ .006

# Read new piecewise linear fit
t_sparse, db_sparse = np.loadtxt(piecewise_model, unpack=True)

# Generate new points for interpolation 
# at same spacing as Peterson model file. 

# If new model starts at end of Peterson (0.1 s), just start from there
if t_sparse.max() == 0.1:
    nsteps = int((np.log10(0.1) - np.log10(t_sparse.min()))/tstep)
    t_new = 10**np.linspace(np.log10(t.min()), 
                            np.log10(t_sparse.min()), nsteps)

# Otherwise, use overlapping Peterson points and then add more at the end
else:
    i_peterson, = np.where(t < t_sparse.max())
    t_peterson = t[i_peterson][:-1] #skip last point at 0.1 s 

    nsteps = int((np.log10(0.1) - np.log10(t_sparse.min()))/tstep)
    log_extend = np.linspace(-1, np.log10(t_sparse.min()), nsteps)
    t_extend = 10**log_extend
    # Concatenate original Peterson model pts and piecewise linear pts
    t_new = np.append(t_peterson, t_extend)


print('min, max period:', t_new.max(), t_new.min())
print('min, max freq:', 1./t_new.max(), 1./t_new.min())

# Interpolate piecewise noise models onto new points.
# need to take log10 so we get a linear fit with log10(period)
# And need to reverse order of all arrays so x-values are increasing
db_new = np.interp(np.log10(t_new)[::-1], np.log10(t_sparse)[::-1], db_sparse[::-1])
db_new = db_new[::-1]

# Write out original Peterson points and new interpolated points
outfile = open(stitch_model, 'w')
for i in range(len(t)):
    if t[i] > t_sparse.max():
        outfile.write('{0:8.6f} {1:8.4f}\n'.format(t[i], db[i]))
for i in range(len(t_new)):
    outfile.write('{0:8.6f} {1:8.4f}\n'.format(t_new[i], db_new[i]))
outfile.close()

