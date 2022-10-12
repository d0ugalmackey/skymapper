## Python script create_fringe_pcs.py -- use a subset of the preprocessed images to create a set of fringe PCs for this amp
## Requires numpy, scipy, astropy, and sklearn

## create_fringe_pcs.py:
## Filters the list of potential input images to create a subset of images with the most appropriate qualities for PC creation
## The images must have been processed with the python script preprocess_images.py
## The logfile must have been reformatted with reformat_logfile.pl
## This script uses the same PCA engine as the bias correction routine bias_magic.py
## Although the defringing problem naively looks different as it is 2-dimensional....
## ... we simply reformat each image into a single very long row and then reassemble things after the PCA is complete.
## The fringe PCs are saved for later use where they will be subtracted in linear combination from a given set of science frames

##### Import packages #####
import numpy as np
import astropy.io.fits as fits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import scipy.stats as stats
from subprocess import call
import scipy.ndimage as ndimage
import time
import warnings

##### Set constants, file locations, and other useful parameters #####
print ("Initialising")

# Note that we don't need to specify the amp as this should propagate from the file lists
# Start timing
start_time = time.time()

# Specify the reformatted list of preprocessed images & statistics
listfile = 'preprocessed_images.dat'

# Parameters for the PCA
nc = 20            # Number of principal components to keep
doscale = 1        # Should we scale the images first (1 = yes)
use32bit = 1       # Using 32-bit arithmetic saves memory, but costs a little bit of precision (turns out not important)
limfrac = 0.00135  # Fraction of distribution to trim on each side when deriving coefficient limits (>~ 3-sigma outliers)

# Filtering parameters to create the best subset for PC creation
# (This took weeks of experimentation to define!)
# Try to strike a balance between keeping a large input set, and having "ideal" characteristics
# First exclude the top and bottom 5% of images in terms of residual median pixel value
mnperc1 = 5
mnperc2 = 95
# Only a small fraction of the pixels should have been masked (throw out crowded fields where there isn't much clean background)
# Also throw out images with very few masked pixels --> no sources --> something is fishy!
fmask1 = 0.02
fmask2 = 0.15
# Really don't want a large number of saturated pixels --> bright stars --> all sorts of problems (especially diffuse scattered light)
fsat1 = 0.000
fsat2 = 0.00075
# Check the original background level -- exclude anything really high or low here as this can indicate complications that we just don't need
back1 = 250.0
back2 = 2500.0
# Be stricter on the exposure time -- only images with texp > 100s
mexp = 100.0

# Set the image size -- by this point in the SDP this is fixed for all images
m = 4096
n = 2048

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for initialising was",elapsed_time,"seconds"
start_time = time.time()

##### Do the filtering #####

# First read in the statistics for all images
exp, mn, sig, fm, fs, bg = np.loadtxt(listfile, usecols=(2,3,4,5,6,7), unpack=True)
imagelist = np.loadtxt(listfile, dtype=np.str, usecols=(0,))
nall = len(imagelist)

# Calculate the percentile values for the median pixel filter
tmp = mn[(exp >= mexp) & (bg >= back1) & (bg <= back2) & (fm >= fmask1) & (fm <= fmask2) & (fs >= fsat1) & (fs <= fsat2)]
mn1 = np.percentile(tmp,mnperc1)
mn2 = np.percentile(tmp,mnperc2)

# Apply the full filter and make a list of the images that pass
imgood = imagelist[(exp >= mexp) & (bg >= back1) & (bg <= back2) & (fm >= fmask1) & (fm <= fmask2) & (fs >= fsat1) & (fs <= fsat2) & (mn >= mn1) & (mn <= mn2)]
ntot = len(imgood)

# Report the results for piping to a log file
print "Preparing to construct fringe PCs, using images in:",listfile
print "Of",nall,"input images,",ntot,"passed the following quality cuts:"
print "Mean residual in the range",mean1,"-",mean2
print "Fraction masked in the range",fmask1,"-",fmask2
print "Fraction saturated in the range",fsat1,"-",fsat2
print "Subtracted background in the range",back1,"-",back2

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for filtering was",elapsed_time,"seconds"
start_time = time.time()

##### Now create the PCs #####

# Set what arithmetic we're going to use and set up the empty data array
if (use32bit == 1):
    print "Using 32-bit arithmetic"
    fringedata = np.empty((ntot,m*n), dtype='float32')
else:
    print "Using 64-bit arithmetic"
    fringedata = np.empty((ntot,m*n), dtype='float64')

# Report on the option of scaling images
# Note that scaling should always be done -- empirically produces much better results for this algorithm
if (doscale == 1):
    print "Will scale all images to zero mean and unit variance before constructing PCs"
else:
    print "Will not scale the input images before constructing PCs"

# Now read in each "good" frame and store in the data array
nn = -1
for q in range(len(imgood)):
    infile = 'preprocessed/' + str(imgood[q]) + '_proc.fits'
    data = fits.getdata(infile, header=False)
    nn = nn + 1

    # If we selected scaling, then do the scaling
    # Note that we explicitly reshape all data here into long rows
    print "Working on usable image #",nn,":",infile
    if (doscale == 1):
        data2 = StandardScaler().fit_transform(data.reshape(1,-1).T)
    else:
        data2 = data.reshape(1,-1)
        
    # Make this a 32-bit frame if selected
    if (use32bit == 1):
        fringedata[nn,:] = data2.astype('float32').reshape(1,-1)
    else:
        fringedata[nn,:] = data2.reshape(1,-1)

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for reading in the images was",elapsed_time,"seconds"
start_time = time.time()

# Now do the PCA
# This needs to be run on raijin as it is very memory intensive
# ~3000 input images requires about 700GB of RAM and 2hr of CPU time *per amp*
print "Running training set PCA - using",ntot,"input images"
pca = PCA(n_components=nc)
coeff = pca.fit_transform(fringedata)
print "Explained variance:"
print (pca.explained_variance_ratio_)

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for PCA solution was",elapsed_time,"seconds"
start_time = time.time()

# Reshape the computed PCs back into the correct shape and write as fits images for inspection
outcode = 's' + str(doscale) + 'a' + str(use32bit) + 'n' + str(ntot)
fringeout = pca.mean_.reshape(m,n)
filename = 'fringe_pc0_' + str(outcode) + '.fits'
warnings.filterwarnings('ignore', category=UserWarning, append=True)
fits.writeto(filename, fringeout, clobber=True)

for j in range(nc):
    fringeout = pca.components_[j].reshape(m,n)
    filename = 'fringe_pc' + str(j+1) + '_' + str(outcode) + '.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, fringeout, clobber=True)

# We also want to know what the typical ranges of the PC coefficients are across the set of input frames
# This will be useful to stop the "correction" running out of control on science frames
# For each of the top nc PCs, form the distribution of linear coefficients from each row
# Trim the very edges of the distribution and save the minimum and maximum values
limits = np.zeros((nc,2))
for j in range(nc):
    cf = np.copy(coeff[:,j])
    cft = stats.trimboth(np.sort(cf),limfrac)
    limits[j,0] = 1.0*np.min(cft)
    limits[j,1] = 1.0*np.max(cft)

# Write out this information for use by the PC subtraction code
filename = 'pc_limits_' + str(outcode) + '.dat'
np.savetxt(filename, limits, fmt='%.10f')
filename = 'pc_coeffs_' + str(outcode) + '.dat'
np.savetxt(filename, coeff, fmt='%.10f')

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for writing output was",elapsed_time,"seconds"

##### That's all! #####
