## Python script bias_magic.py -- learn the nightly PCs describing row-oriented noise patterns in SkyMapper images
## Requires numpy, scipy, sklearn and astropy

## Some brief notes on SkyMapper images:
## The focal plane of the SkyMapper camera is tiled with 32 CCDs containing 2048x4096 pixels each (268.4 Mpix total).
## Each CCD is read out through 2 amplifiers, with the split at column 1024. Hence a given image has 64 extensions ("frames").
## Since the 2 amps read in opposite directions, half of the frames need their x-coordinates flipped before analysis.
## During readout, an extra 50 pix before and after the data are also recorded ("pre-scan" and "post-scan").
## The idea behind this is to allow the current base level of the electronics to be corrected.
## This is typically set to be significantly non-zero to eliminate negative pixel values.
## At the beginning and end of a given night, a set of ~ten "bias" images are saved.
## These have zero exposure time (i.e., are readout only), recording the row-oriented noise patterns for that night.
## SkyMapper images are stored in the astronomical standard fits format, see https://en.wikipedia.org/wiki/FITS

## bias_magic.py:
## Uses the bias frames to learn the main principal components (PCs) describing the noise along a row on a given night.
## These are saved for use by science_magic.py, which elimates the noise pattern from science images taken on that night.
## The general procedure is to, for each amp, create the median bias for the night and subtract it from each individual frame.
## Then mask any deviant pixels and learn the PCs from the set of all rows except the first ten on each frame.


##### Import packages #####
import numpy as np
import astropy.io.fits as fits
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import scipy.stats as stats
import warnings

##### Set constants and other useful parameters #####
print ("Initialising")

# Suppress warnings about NaNs (there can be lots)
np.seterr(invalid='ignore')

# Data related constants
namp = 64   # Number of amps

# Parameters for trimming of pre- and post-scan, and subtraction of the post-scan level
cut1 = 50   # Width of pre-scan in pixels
cut2 = 50   # Width of post-scan in pixels
os = 40     # Width of post-scan region used to determine mean level

# Parameters for robust bad-pixel masking
# Step 1: crudely trim the distribution of pixel values and calculate median and sigma from this modified distribution
# Step 2: trim any pixels from the frame that sit outside some number of sigma (+/-) from the median
bpfrac = 0.025        # Step 1: fraction of the Gaussian part of the pixel value distribution to trim each side (>~ 2-sigma outliers)
extrm = 35            # Step 1: also cut any pixels with "extreme" values +/-
bp1, bp2 = 4.5, 3.5   # Step 2: trim pixels sitting outside +bp1 and -bp2 sigma from the median

# Parameters for the PCA
trimend = 10       # How many rows to ignore at the bottom of each frame (empirically, these can have additional noise)
nc = 10            # Number of principal components to keep
limfrac = 0.00135  # Fraction of distribution to trim on each side when deriving coefficient limits (>~ 3-sigma outliers)

# Set file names 
biasfile = "biaslist.dat"     # List of input bias images (created by the helper script or manually)
scifile = "sciencelist.dat"   # List of input science images (created by the helper script or manually)
medfile = "medbias.fits"      # Fits file containing the median bias image (64 extensions)

# Figure out how many bias images there are
biaslist = np.loadtxt(biasfile, dtype=np.str)
nbias = len(np.atleast_1d(biaslist))

# Figure out how many science images there are
scilist = np.loadtxt(scifile, dtype=np.str)
nsci = len(np.atleast_1d(scilist))

# Figure out the size of a single raw frame (can be variable at this point in the SDP in rare cases)
infile = biaslist[0]                               # Select the first bias image in the list
data, hdr = fits.getdata(infile, 1, header=True)   # Read the first extension; strip the fits header
m,n = data.shape

# Figure out where the edges of the pre- and post-scan are (column indices)
edge1 = cut1
edge2 = n - cut2

# Figure out the size of a single trimmed frame
trim = np.copy(data[:,edge1:edge2])
mt,nt = trim.shape

##### Construct the median bias for all amps #####
print ("Constructing median frames for",namp,"amps")

# Create the data arrays that will hold the individual bias images and the median biases
biasdata = np.zeros((namp,nbias,mt,nt))
medbias = np.zeros((namp,mt,nt))

# Loop over all amps
for nn in range(namp):
    amp = nn+1

    # Loop over all bias frames
    for q in range(nbias):

        # Read in the data
        infile = biaslist[q]
        data, hdr = fits.getdata(infile, amp, header=True)     # Make sure to strip the fits header

        # Check to see if the frame needs to be flipped in x
        if amp==1 or amp==3 or amp==5 or amp==7 or amp==9 or amp==11 or amp==13 or amp==15 or amp==17 or amp==19 or amp==21 or amp==23 or amp==25 or amp==27 or amp==29 or amp==31 or amp==34 or amp==36 or amp==38 or amp==40 or amp==42 or amp==44 or amp==46 or amp==48 or amp==50 or amp==52 or amp==54 or amp==56 or amp==58 or amp==60 or amp==62 or amp==64:
            data = np.fliplr(data)

        # Row by row, calculate the post-scan level (use median) and subtract from the trimmed data
        osmn = np.median(np.copy(data[:,edge2:(edge2+os)]),axis=1)
        trim = np.copy(data[:,edge1:edge2])
        tsub = np.zeros((mt,nt))
        for j in range(mt):
            tsub[j,:] = np.copy(trim[j,:]) - osmn[j]

        # Store the subtracted frame in the data array
        biasdata[nn,q,:,:] = tsub

    # For the current amp, calculate the median of all the bias frames
    medbias[nn,:,:] = np.median(np.copy(biasdata[nn,:,:,:]), axis=0)

    # Populate a fits file with the median frames (64 extensions)
    # We are writing this to disk as it will be required later by science_magic.py
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    if (amp == 1):
        fits.writeto(medfile, np.copy(medbias[nn,:,:]), clobber=True)   # If this is the first amp then create the file
    else:
        fits.append(medfile, np.copy(medbias[nn,:,:]))   # Otherwise, just append another extension for this amp
    
##### Subtract the median frames from all biases and mask out deviant pixels #####
print ("Subtracting median frames from all biases and masking deviant pixels")

# Loop over all amps
for nn in range(namp):
    amp = nn+1

    # Loop over all bias frames
    for q in range(nbias):

        # Subtract the median frame -- these biases should now have pixel values roughly centred on zero
        biasdata[nn,q,:,:] = np.copy(biasdata[nn,q,:,:]) - np.copy(medbias[nn,:,:])

        # Mask out deviant pixels by trimming either end of the distribution of pixel values
        # Distribution generally looks Gaussian but the wings can be strange, stacked with high or low pixels
        # To accurately measure the median and variance we clip outliers (default >~ 2-sigma) from the Gaussian part...
        # ... plus anything with "extreme" pixel values (the default +/- 35 counts is conservative)
        frac1 = bpfrac + (1.0*np.sum(np.copy(biasdata[nn,q,:,:]) < -1.0*extrm)) / (mt*nt)
        frac2 = bpfrac + (1.0*np.sum(np.copy(biasdata[nn,q,:,:]) > 1.0*extrm)) / (mt*nt)
        tmptmp = stats.trim1(np.sort(np.copy(biasdata[nn,q,:,:]).reshape(1,-1)).T,frac1,tail='left')
        tmp = stats.trim1(tmptmp,frac2,tail='right')

        # Next calculate median and sigma from this modified pixel value distribution
        # Then mask any pixels in the full distribution that sit outside +bp1-sigma, -bp2-sigma from the median
        # In the SDP these values are logged and checked so that any deviant biases can be flagged
        # These can indicate problems with the detector (e.g., unexpected warming) or electronics on a given night.
        tbkg = np.median(tmp)
        tsig = np.std(tmp,dtype=np.float64)
        biasdata[nn,q,:,:][biasdata[nn,q,:,:] > (tbkg+bp1*tsig)] = np.nan
        biasdata[nn,q,:,:][biasdata[nn,q,:,:] < (tbkg-bp2*tsig)] = np.nan        

##### Determine the principal components from the bias training set #####
print ("Determining the principal components from the bias training set")

# Create the arrays for storing the PC solutions and coefficient limits
princomp = np.zeros((namp,nt,nc+1))
pclimits = np.zeros((namp,nc,2))

# Loop over all amps
for nn in range(namp):
    amp = nn+1

    # Initialise the row counter
    ntot = 0

    # Work through all bias frames for this amp to create the training set
    # Never accept a row right at the bottom of a frame -- often looks weird (trimend = 10 is very conservative)
    # Loop over all bias frames
    for q in range(nbias):
        
        ngood = mt - trimend
        tsetp = np.zeros((ngood,nt))
        kk = 0
        for j in range(mt):
            if j >= trimend:
                tsetp[kk,:] = np.copy(biasdata[nn,q,j,:])
                kk += 1
                
        ntot += kk

        # Now append the list of good rows onto the overall list for this amp (if q=0 then create)
        if q == 0:
            tset = np.copy(tsetp)
        else:
            tset = np.concatenate((tset,tsetp))

    # Use the sklearn imputer to fill in the NaNs with the column median (recall bad pixels were masked with NaN)
    # Removing the bad pixels like this stops the PCs from being affected by very deviant values in the training set
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(tset)       
    tset2 = imp.transform(tset)
            
    # Now run the PCA routine from sklearn
    # This returns the top nc PCs, as well as the linear coefficients for every row
    # (Note that every row can be described by a linear combination of the nc PCs)
    pca = PCA(n_components=nc)
    coeff = pca.fit_transform(tset2)

    # Print the results of the PCA to file for use in the science correction routine science_magic.py
    # The first column will be pca.mean_ and the others will be the PCs (1 ... N)
    results = np.concatenate((pca.mean_.reshape(1,-1),pca.components_)).T
    filename = 'pc_solution_amp' + str(amp) + '_pc' + str(nc) + '.dat'
    np.savetxt(filename, results, fmt='%.10f')

    # For each of the top nc PCs, form the distribution of linear coefficients from each row
    # Trim the very edges of the distribution and save the minimum and maximum values
    # These will serve as a guide to stop the "correction" running out of control on science frames
    # Print the coefficient limits to file for use by science_magic.py
    limits = np.zeros((nc,2))
    for j in range(nc):
        cf = np.copy(coeff[:,j])
        cft = stats.trimboth(np.sort(cf),limfrac)
        limits[j,0] = 1.0*np.min(cft)
        limits[j,1] = 1.0*np.max(cft)

    filename = 'pc_limits_amp' + str(amp) + '_pc' + str(nc) + '.dat'
    np.savetxt(filename, limits, fmt='%.10f')

    # Save the PCs and limits into the relevant arrays
    princomp[nn,:,:] = np.copy(results)
    pclimits[nn,:,:] = np.copy(limits)

# Delete the huge array with all the bias frames in it
# Not 100% sure if this releases the memory, but we don't need the array again anyway
del biasdata

##### That's all! #####
