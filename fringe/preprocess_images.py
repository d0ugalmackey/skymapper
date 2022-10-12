## Python script preprocess_images.py -- run preprocessing on potential source images for fringe PC creation
## Requires numpy, scipy, astropy, and a locally-installed instance of Source Extractor
## For the latter see: https://www.astromatic.net/software/sextractor/

## preprocess_images.py:
## Clean up each potential input image - mask bad pixels, all sources; subtract background
## Perform and record some statistics on the cleaned frame
## This info will let us select a final input list of high quality frames to create the fringe PCs
## Do this for a single amp 

##### Import packages #####
import numpy as np
import astropy.io.fits as fits
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

# Suppress warnings about NaNs
np.seterr(invalid='ignore')

# Which amp are we looking at?
amp = 17

# Parameters for robust bad-pixel masking
# Step 1: crudely trim the distribution of pixel values and calculate median and sigma from this modified distribution
# Step 2: trim any pixels from the frame that sit outside some number of sigma (+/-) from the median
bpfrac = 0.025        # Step 1: fraction of the Gaussian part of the pixel value distribution to trim each side (>~ 2-sigma outliers)
bp1, bp2 = 4.5, 3.5   # Step 2: trim pixels sitting outside +bp1 and -bp2 sigma from the median

# Set the input list and the source directory for the images
listfile = 'input_list.dat'
basepath = '/priv/manta2/skymap/sdp/edr_saves/reduced_data/'

# Get the list of images, dates, and figure out how many input images there are
mjd = np.loadtxt(listfile, dtype=np.int, usecols=(0,))
imagelist = np.loadtxt(listfile, dtype=np.str, usecols=(1,))
print "Preparing frames for constructing fringe PCs, using images in",listfile
ntot = len(imagelist)

# Figure out the size of a single frame
infile = str(basepath) + str(mjd[0]) + '/' + str(amp) + '/' + str(imagelist[0]) + '.fits'
data, hdr = fits.getdata(infile, header=True)
m,n = data.shape

# Timing check
elapsed_time = time.time() - start_time
print "Elapsed time for initialising was",elapsed_time,"seconds"
start_time = time.time()

##### Run the preprocessing for all images in the list #####
print ("Processing images for amp",amp)

# Read in each frame, run Source Extractor, mask stars and subtract background, and store in data array
for q in range(len(imagelist)):
    inbase = imagelist[q]
    date = mjd[q]
    infile = str(basepath) + str(date) + '/' + amp + '/' + str(inbase) + '.fits'

    data, hdr = fits.getdata(infile, header=True)
    
    # Try and mask any pixel values that are NaN or inf
    data[np.isnan(data)] = 60000.0
    data[np.isinf(data)] = -60000.0

    # Check to see if this is a decent image by calculating the fraction of "saturated" pixels
    # Here this means pixels with very high or low values
    fsat = (1.0*np.sum(data > 40000.0) + 1.0*np.sum(data < -40000.0)) / (m*n)

    # Write an image for source extractor object masking and background map
    filename = 'sxtractor.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, data, clobber=True)
    
    # Run source extractor to create object mask and background map
    # The parameters controlling object detection and background creation are in fringe_correct.sxt
    call(['/priv/manta2/skymap/sdp/deploy/external/sxt','sxtractor.fits','-c','fringe_correct.sxt'])
    
    # Read in the background map and subtract from the working image
    backgr, hdr = fits.getdata('background.fits', header=True)
    bsub = np.copy(data) - backgr
    
    # Read in the object mask and convolve with a Gaussian kernel to broaden the regions for masking
    # This conservatively ensures that for each object the masked area is somewhat larger than the source itself
    obj, hdr = fits.getdata('objects.fits', header=True)
    objmask = ndimage.gaussian_filter(obj, sigma=1.5, order=0)

    # Zero-out all detected sources in the frame by setting masked pixels to large negative value
    # The SDP also uses the static amp-specific bad-pixel mask here as well
    tmask = np.copy(bsub)
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
    tmask[tmask > (tbkg+bp1*tsig)] = tbkg   # Replace masked pixels with the median background value ~ 0
    tmask[tmask < (tbkg-bp2*tsig)] = tbkg
    fmask = frac1 - 0.025                   # Total fraction of pixels that were masked

    # Calculate the median level for the smoothed source extractor background image
    frac1 = bpfrac
    frac2 = bpfrac
    tmptmp = stats.trim1(np.sort(backgr.reshape(1,-1)).T,frac1,tail='left')
    tmp = stats.trim1(tmptmp,frac2,tail='right')
    tbkg2 = np.median(tmp)

    # Record the image stats
    # These will be used to select a subset of the best images for PC creation
    # This is piped to a log file on the command line when this script is executed
    print "Median residual, sigma, fmask, fsat, background (",inbase,"):",tbkg,tsig,fmask,fsat,tbkg2

    # Write out the processed image so it's there if/when we need it
    filename = str(inbase) + '_proc.fits'
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    fits.writeto(filename, tmask, clobber=True)

# Final timing check
elapsed_time = time.time() - start_time
print "Elapsed time for image preparation was",elapsed_time,"seconds"
start_time = time.time()

print "Processed and wrote",ntot,"images"

##### That's all! #####
