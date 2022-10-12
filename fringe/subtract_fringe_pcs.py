## Python script create_fringe_pcs.py -- use a subset of the preprocessed images to create a set of fringe PCs for this amp
## Requires numpy, scipy, astropy, sklearn and a local installtion of Source Extractor (as usual)

## subtract_fringe_pcs.py:
## Uses the main principal components (PCs) describing the fringe pattern to remove fringes from science frames.
## The general procedure is to follow exactly the same pre-processing as was done with the training set (preprocess_images.py)
## That is, mask bad pixels and then use Source Extractor to:
## (i) mask all pixels associated with astronomical sources
## (ii) create a highly smoothed background image that is then subtracted
## Note this second step eliminates large scale gradients or patterns so that these don't get fit by the PCA routine.
## Then reformat the image into a long row, fit linear combination of the PCs to model the fringe pattern, and subtract.

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

# Start timing
start_time = time.time()

# Specify the reformatted list of preprocessed images & statistics
# Note that we want to apply the defringing to *all* images in the set (not just the training images)
listfile = 'preprocessed_images.dat'

# Where do the original images live?
basepath = '/Users/dougal/Astronomy/SkyMapper/fringe/all/z-band/manta/'

# Parameters for robust bad-pixel masking
# Step 1: crudely trim the distribution of pixel values and calculate median and sigma from this modified distribution
# Step 2: trim any pixels from the frame that sit outside some number of sigma (+/-) from the median
bpfrac = 0.025        # Step 1: fraction of the Gaussian part of the pixel value distribution to trim each side (>~ 2-sigma outliers)
bp1, bp2 = 4.5, 3.5   # Step 2: trim pixels sitting outside +bp1 and -bp2 sigma from the median

# Parameters for the PCA
nc = 20            # Number of principal components that were computed
nuse = 20          # Number of PCs to use in the subtraction (<= nc)
doscale = 1        # Should we scale the images first (1 = yes)
use32bit = 1       # Using 32-bit arithmetic saves memory, but costs a little bit of precision (turns out not important)
ntr = 2889         # Number of training images used to compute the PCs
pcbase = 's' + str(doscale) + 'a' + str(use32bit) + 'n' + str(ntr)     # Specifies files where the PC images were stored

# Set the image size -- by this point in the SDP this is fixed for all images
m = 4096
n = 2048

# Load the list of input images and figure out how many we have to fix
imagelist = np.loadtxt(listfile, dtype=np.str, usecols=(0,))
nsci = len(np.atleast_1d(imagelist))

# Create data arrays for the fringe PCs
if (use32bit == 1):
    print "Using 32-bit arithmetic"
    fringepc = np.empty((m*n,nuse), dtype='float32')
else:
    print "Using 64-bit arithmetic"
    fringepc = np.empty((m*n,nuse), dtype='float64')

# Read in the fringe PCs and reshape into rows
# Mean first
filename = 'fringe_pc0_' + str(code) + '.fits'
tmp, hdr = fits.getdata(filename, header=True)
if (use32bit == 1):
    fringe_mean = tmp.astype('float32').reshape(1,-1)
else:
    fringe_mean = tmp.reshape(1,-1)

# Now the other components
for nn in range(nuse):
    ctr = nn + 1
    filename = 'fringe_pc' + str(ctr) + '_' + str(code) + '.fits'
    tmp, hdr = fits.getdata(filename, header=True)
    if (use32bit == 1):
        fringepc[:,nn] = tmp.astype('float32').reshape(1,-1)
    else:
        fringepc[:,nn] = tmp.reshape(1,-1)

# Read in the coefficient limits file
limfile = 'pc_limits_' + str(code) + '.dat'
pclimits = np.loadtxt(limfile)

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for initialising was",elapsed_time,"seconds"
start_time = time.time()

##### Now subtract the fringes #####

# Read in each science frame in the list one by one and do the correction 
for q in range(nsci):
    if (nsci == 1):
        inbase = imagelist[()]
    else:
        inbase = imagelist[q]
    print "Working on image",inbase
    infile = str(basepath) + str(inbase) + '.fits'

    data, hdr = fits.getdata(infile, header=True)

    # Make a copy for working
    data2 = np.copy(data)

    # Now we process the images just like in preprocess_images.py
    # Try and hide any NaNs and infs
    data2[np.isnan(data2)] = 60000.0
    data2[np.isinf(data2)] = -60000.0
    
    # Write an image for source extractor object masking and background map
    filename = 'sxtractor.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, data2, clobber=True)

    # Run source extractor to create object mask and background map
    call(['sxt','sxtractor.fits','-c','fringe_correct.sxt'])

    # Read in the background map and subtract from the working image
    backgr, hdr = fits.getdata('background.fits', header=True)
    bsub = np.copy(data2) - backgr
    
    # Read in the object mask and convolve with a Gaussian kernel to broaden the regions for masking
    # This conservatively ensures that for each object the masked area is somewhat larger than the source itself
    obj, hdr = fits.getdata('objects.fits', header=True)
    objmask = ndimage.gaussian_filter(obj, sigma=1.5, order=0)

    # Zero-out all detected sources in the frame by setting masked pixels to large negative value
    # The SDP also uses the static amp-specific bad-pixel mask here as well
    # We also initialise a second version of the masked image here for later use
    tmask = np.copy(bsub)
    tmask2 = np.copy(tmask)
    tmask[tobjmask > 0.0] = -9999.9  # Mask objects
    
    # Remove any remaining deviant pixels by trimming either end of the distribution of pixel values
    # Distribution generally looks Gaussian (sky background) but with large extension to low values (masked objects)
    # To accurately measure the median and variance we clip outliers (default >~ 2-sigma) from the Gaussian part...
    # ... plus all the source pixels from the object masking
    frac1 = bpfrac + (1.0*np.sum(tobjmask > 0.0)) / (m*n)
    frac2 = bpfrac
    tmptmp = stats.trim1(np.sort(tmask.reshape(1,-1)).T,frac1,tail='left')
    tmp = stats.trim1(tmptmp,frac2,tail='right')

    # Next calculate median and sigma from this modified pixel value distribution
    # Then mask any pixels in the full distribution that sit outside +bp1-sigma, -bp2-sigma from the median
    # In the SDP these values are logged and checked so that any deviant frames can be flagged
    # These can indicate problems with the electronics, or the weather (high background often == clouds, haze, smoke)
    tbkg = np.median(tmp)
    tsig = np.std(tmp,dtype=np.float64)
    # The array tmask is used for object masking so flag with very negative values
    tmask[tmask > (tbkg+bp1*tsig)] = -9999.9
    tmask[tmask < (tbkg-bp2*tsig)] = -9999.9
    # The array tmask2 is used for the standardscaler so replace pixels with the mean background value
    tmask2[tmask > (tbkg+bp1*tsig)] = tbkg  
    tmask2[tmask < (tbkg-bp2*tsig)] = tbkg
    fmask = frac1 - bpfrac                  # Total fraction of pixels that were masked
    
    # Do scaling if necessary (uses tmask2)
    if (doscale == 1):
        scaler = StandardScaler().fit(tmask2.reshape(1,-1).T)
        tmp = scaler.transform(bsub.reshape(1,-1).T)
    else:
        tmp = np.copy(bsub)

    # Convert to 32-bit if necessary (includes reshaping)
    if (use32bit == 1):
        bsub2 = tmp.astype('float32').reshape(1,-1)
    else:
        bsub2 = tmp.reshape(1,-1)        
    
    # Subtract the mean fringe pattern and reshape the object mask
    bsub3 = bsub2 - fringe_mean
    tmask3 = tmask.reshape(1,-1)

    # Timing check
    elapsed_time = time.time() - start_time
    print " - Elapsed time for pre-processing was",elapsed_time,"seconds"
    start_time = time.time()

    # Do a weighted least squares solution to determine the PC coefficients
    coeff = np.zeros(nuse)
    Xp = np.copy(fringepc)
    Xp[tmask3[0,:] == -9999.9] = 1e-10
    Yp = np.copy(bsub3[0,:])
    Yp[tmask3[0,:] == -9999.9] = 1e-10
    coeff = np.linalg.lstsq(Xp,Yp)[0]

    # Check to see how many of the coefficients exceed the limits derived from the training set
    # A large number can indicate a problem; this is logged and checked in the SDP
    for k in range(nuse):
        if (coeff[k] < pclimits[k,0]) or (coeff[k] > pclimits[k,1]):
            print "Solution: component",k,"exceeded limits:",coeff[k],"[",pclimits[k,0],",",pclimits[k,1],"]"

    # Empirically, we never need to worry about these for the fringing -- just check and record the number

    # "Reconstruct" the fringe model from the linear combination of PCs and add the mean vector back on
    recon = np.dot(coeff.T,fringepc.T) + fringe_mean

    # Undo the scaling if necessary
    if (doscale == 1):
        recon2 = scaler.inverse_transform(recon)
    else:
        recon2 = np.copy(recon)
    
    # Subtract the fringe model from the original image
    corr = data.reshape(1,-1) - recon2

    # Now reshape these frames back to the correct dimensions
    corr2 = corr.reshape(m,n)
    recon2 = recon2.reshape(m,n)

    # Make sure everything is still 32-bit if necessary
    if (use32bit == 1):
        corr3 = corr2.astype('float32')
        recon3 = recon2.astype('float32')
    else:
        corr3 = np.copy(corr2)
        recon3 = np.copy(recon2)    
    
    # Timing check
    elapsed_time = time.time() - start_time
    print " - Elapsed time for calculating solution was",elapsed_time,"seconds"
    start_time = time.time()

    # Create output fits files showing the corrected frame and the fringe model for this frame
    filename = str(inbase) + '_fix_' + str(nuse) + '.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, corr3, clobber=True)
    
    filename = str(inbase) + '_frng_' + str(nuse) + '.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, recon3, clobber=True)

    # Timing check
    elapsed_time = time.time() - start_time
    print " - Elapsed time for writing output was",elapsed_time,"seconds"
    start_time = time.time()

##### That's all! #####











