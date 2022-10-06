## Python script science_magic.py -- use the nightly PCs to correct bias noise in all SkyMapper science images from that night
## Requires numpy, scipy, sklearn, astropy, and a locally-installed instance of Source Extractor
## For the latter see: https://www.astromatic.net/software/sextractor/

## See bias_magic.py for information on SkyMapper images

## science_magic.py:
## Uses the main principal components (PCs) describing the noise along a row on a given night to remove noise from science frames.
## The general procedure is to follow exactly the same pre-processing as was done with the bias training set.
## That is, subtract the overscan level and the median bias. Then use Source Extractor to:
## (i) mask all pixels associated with astronomical sources
## (ii) create a highly smoothed background image that is then subtracted
## Note this second step eliminates large scale gradients or patterns so that these don't get fit by the PCA routine.
## Then fit the PCs to each row individually to model the noise and remove it.

##### Import packages #####
import numpy as np
import astropy.io.fits as fits
from sklearn.decomposition import PCA
from subprocess import call
import scipy.stats as stats
import scipy.ndimage as ndimage
import warnings

##### Set constants and other useful parameters #####
print ("Initialising")

# Suppress warnings about NaNs
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
bp1, bp2 = 4.5, 3.5   # Step 2: trim pixels sitting outside +bp1 and -bp2 sigma from the median

# Parameters for the PCA
nc = 10            # Number of principal components to keep

# Parameters for checking the PC coefficients in the derived solutions
enforcelimits = 1   # Enforce the empirical coefficient limits when applying the PC correction to science?  (1=yes)
lmult = 1.25        # Multiplier for relaxing the limits

# Set initial file names
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

##### Read in and store the median biases and PC solutions computed by bias_magic.py #####
print ("Reading median bias frames and PC solutions")
medbias = np.zeros((namp,mt,nt))
princomp = np.zeros((namp,nt,nc+1))
pclimits = np.zeros((namp,nc,2))

# Loop over all amps
for nn in range(namp):
    amp = nn+1

    # Read in the median bias for this amp
    data, hdr = fits.getdata(medfile, nn, header=True)     # Make sure to strip the fits header
    medbias[nn,:,:] = np.copy(data)

    # Read in the PCs for for this amp
    infile = 'pc_solution_amp' + str(amp) + '_pc' + str(nc) + '.dat'
    pcsol = np.loadtxt(infile)
    princomp[nn,:,:] = np.copy(pcsol)

    # Read in the linear coefficient limits for this amp
    infile = 'pc_limits_amp' + str(amp) + '_pc' + str(nc) + '.dat'
    pcsol2 = np.loadtxt(infile)
    pclimits[nn,:,:] = np.copy(pcsol2)

##### Now apply the PCA solutions to the science frames to remove the bias pattern #####
print ("Applying PCA correction to science frames")

# Loop over all science frames
for q in range(nsci):
    if (nsci == 1):
        inbase = scilist[()]
    else:
        inbase = scilist[q]

    infile = str(inbase) + '.fits'
    print (infile)
        
    # Loop over all amps
    for nn in range(namp):
        amp = nn+1

        # Read in the data
        data, hdr = fits.getdata(infile, amp, header=True)     # Make sure to strip the fits header

        # Check to see if the frame needs to be flipped in x
        if amp==1 or amp==3 or amp==5 or amp==7 or amp==9 or amp==11 or amp==13 or amp==15 or amp==17 or amp==19 or amp==21 or amp==23 or amp==25 or amp==27 or amp==29 or amp==31 or amp==34 or amp==36 or amp==38 or amp==40 or amp==42 or amp==44 or amp==46 or amp==48 or amp==50 or amp==52 or amp==54 or amp==56 or amp==58 or amp==60 or amp==62 or amp==64:
            data = np.fliplr(data)

        # Row by row, calculate the post-scan level (use median) and subtract from the trimmed data
        osmn = np.median(np.copy(data[:,edge2:(edge2+os)]),axis=1)
        trim = np.copy(data[:,edge1:edge2])
        trim2 = np.zeros((mt,nt))
        for j in range(mt):
            trim2[j,:] = np.copy(trim[j,:]) - osmn[j]

        # Subtract the median frame
        trim3 = np.zeros((mt,nt))
        trim3 = np.copy(trim2) - np.copy(medbias[nn,:,:])

        # Write an image for Source Extractor object masking and background map
        filename = 'sxtractor.fits'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        fits.writeto(filename, trim3, clobber=True)

        # Run Source Extractor to create the object mask and background map
        # The parameters controlling object detection and background creation are in bias_correct.sxt
        call(['/priv/manta2/skymap/sdp/deploy/external/sxt','sxtractor.fits','-c','bias_correct.sxt'])

        # Read in the background map and subtract from the working image
        backgr, hdr = fits.getdata('background.fits', header=True)
        tsub = np.copy(trim3) - np.copy(backgr)

        # Read in the object mask and convolve with a Gaussian kernel to broaden (slightly) the regions for masking
        # This conservatively ensures that for each object the masked area is a little larger than the source itself
        obj, hdr = fits.getdata('objects.fits', header=True)
        tobjmask = ndimage.gaussian_filter(obj, sigma=0.5, order=0)

        # Zero-out all detected sources in the frame by setting masked pixels to large negative value
        # The SDP also uses the static amp-specific bad-pixel mask here as well
        tmask = np.copy(trim3)
        tmask[tobjmask > 0.0] = -9999.9  # Mask objects

        # Remove any remaining deviant pixels by trimming either end of the distribution of pixel values
        # Distribution generally looks Gaussian (sky background) but with large extension to low values (masked objects)
        # To accurately measure the median and variance we clip outliers (default >~ 2-sigma) from the Gaussian part...
        # ... plus all the source pixels from the object masking
        frac1 = bpfrac + (1.0*np.sum(tobjmask > 0.0)) / (mt*nt)
        frac2 = bpfrac
        tmptmp = stats.trim1(np.sort(tmask.reshape(1,-1)).T,frac1,tail='left')
        tmp = stats.trim1(tmptmp,frac2,tail='right')

        # Next calculate median and sigma from this modified pixel value distribution
        # Then mask any pixels in the full distribution that sit outside +bp1-sigma, -bp2-sigma from the median
        # In the SDP these values are logged and checked so that any deviant frames can be flagged
        # These can indicate problems with the electronics, or the weather (high background often == clouds, haze, smoke)
        tbkg = np.median(tmp)
        tsig = np.std(tmp,dtype=np.float64)
        tmask[tmask > (tbkg+bp1*tsig)] = -9999.9  # This will capture any remaining hot pixels
        tmask[tmask < (tbkg-bp2*tsig)] = -9999.9  # Mask low pixel values too

        # Make the correction using the inverse PC transformation
        # First obtain the relevant slices of the PC solution for this amp
        pca_mean = np.copy(princomp[nn,:,0])
        pca_components = np.copy(princomp[nn,:,1:nc+1])
        pclim = np.copy(pclimits[nn,:,:])

        # Subtract the mean vector from each column
        tsub2 = np.zeros((mt,nt))
        for j in range(nt):
            tsub2[:,j] = np.copy(tsub[:,j]) - pca_mean[j]

### This next section of code is mathematically elegant but *REALLY* slow, mainly due to repeated use of np.dot.
### It's so slow the SDP could not run through all the SkyMapper data in a tractable time (would be > 1 year!)
### Excised and replaced with the subsequent code, which is a bit hacky but *at least* ten times faster.
#            
#        # Construct a weight mask such that any masked pixel gets ~zero weight in the transform
#        weights = np.ones(tsub2.shape)
#        weights[tmask == -9999.9] = 1e-10
#        weights[tmask == -9999.9] = 1e-10
#
#        # Do a set of weighted least squares solutions to determine the PC coefficients row by row
#        # If a high fraction of pixels in a row are masked (large object or crowded field) then fit commensurately fewer PCs
#        # In the worst case (rare) do not attempt a fit
#        coeff = np.zeros((nc,mt))
#        for j in range(mt):
#            cfrac = 1.0*np.sum(weights[j,:] > 1e-10) / nt   # Fraction of non-masked pixels in the row
#
#            if cfrac >= 0.25:    
#                # If the fraction of good pixels is higher than 25% then fit all the PCs
#                w = np.diag(np.copy(weights[j,:]))
#                Xp = np.dot(w,pca_components)
#                Yp = np.dot(w,np.copy(tsub2[j,:]))
#                beta = np.linalg.lstsq(Xp,Yp)[0]
#                coeff[:,j] = np.copy(beta)
#        
#            elif (cfrac < 0.25) and (cfrac >= 0.10):
#                # If the fraction is 10-25% then fit only the first 5 PCs
#                pcacomp = np.copy(pca_components)
#                pcacomp[:,5:nc] = np.zeros((nt,nc-5))
#                w = np.diag(np.copy(weights[j,:]))
#                Xp = np.dot(w,pcacomp)
#                Yp = np.dot(w,np.copy(tsub2[j,:]))
#                beta = np.linalg.lstsq(Xp,Yp)[0]
#                coeff[:,j] = np.copy(beta)
#        
#            elif (cfrac < 0.10) and (cfrac >= 0.02):
#                # If the fraction is 2-10% then fit only the first PC
#                pcacomp = np.copy(pca_components)
#                pcacomp[:,1:nc] = np.zeros((nt,nc-1))
#                w = np.diag(np.copy(weights[j,:]))
#                Xp = np.dot(w,pcacomp)
#                Yp = np.dot(w,np.copy(tsub2[j,:]))
#                beta = np.linalg.lstsq(Xp,Yp)[0]
#                coeff[:,j] = np.copy(beta)
#
#            else:
#                # If there's fewer than 2% good pixels then don't fit at all
#                coeff[:,j] = np.zeros(nc)
#
### ---------------------------------------------------------------------------------------------------------------------

        # Do a set of weighted least squares solutions to determine the PC coefficients row by row
        # If a high fraction of pixels in a row are masked (large object or crowded field) then fit commensurately fewer PCs
        # If the fraction of good pixels is higher than 25% then fit all ten PCs
        # If the fraction is 10-25% then fit only the first 5 PCs
        # If the fraction is 2-10% then fit only the first PC
        # In the worst case (rare) do not attempt a fit
        coeff = np.zeros((nc,mt))
        for j in range(mt):
            x = np.array([1.0*np.sum(tmask[j,:] > -9999.9) / nt])
            nr = np.asscalar(np.piecewise(x, [x >= 0.25, (x < 0.25) & (x >= 0.10), (x < 0.10) & (x >= 0.02)], [0,5,9,10]))

            Xp = np.copy(pca_components)
            Xp[:,nc-nr:nc] = np.zeros((nt,nr))
            Xp[tmask[j,:] == -9999.9] = 1e-10                # Give ~zero weight to masked pixels
            tsub2[j,:][tmask[j,:] == -9999.9] = 1e-10
            coeff[:,j] = np.linalg.lstsq(Xp,tsub2[j,:])[0]            
                
        # Check to see how many of the rows exceed the limits derived from the training set
        # There is usually a handful per frame, but a large number can indicate a problem
        # This is logged and checked in the SDP
        nbad = 0
        for j in range(mt):
            for k in range(nc):
                lim = lmult*np.maximum(np.absolute(pclim[k,0]),np.absolute(pclim[k,1]))
                if (coeff[k,j] < -lim) or (coeff[k,j] > lim):
                    nbad += 1
            
        # If desired, trim outlying coefficients to conform to the limits
        # Generally this doesn't seem to be necessary, but is enforced in the SDP to ensure QC
        if enforcelimits == 1:
            for k in range(nc):
                tc = np.copy(coeff[k,:]).reshape(1,-1)
                lim = lmult*np.maximum(np.absolute(pclim[k,0]),np.absolute(pclim[k,1]))
                tc[tc < -lim] = -lim
                tc[tc > lim] = lim
                if k == 0:
                    tcoeff = np.copy(tc)
                else:
                    tcoeff = np.concatenate((tcoeff,tc))
            coeff = np.copy(tcoeff)

        # Construct the noise model for this frame row by row using the linear combination of PCs, and add the mean vector back on. 
        reconsub = np.dot(coeff.T,pca_components.T)
        recon = np.zeros((mt,nt))
        for j in range(nt):
            recon[:,j] = np.copy(reconsub[:,j]) + pca_mean[j]

        # Subtract the noise model from the image
        corr = trim3 - recon

        # Reverse the data if necessary (restore the original direction)
        if amp==1 or amp==3 or amp==5 or amp==7 or amp==9 or amp==11 or amp==13 or amp==15 or amp==17 or amp==19 or amp==21 or amp==23 or amp==25 or amp==27 or amp==29 or amp==31 or amp==34 or amp==36 or amp==38 or amp==40 or amp==42 or amp==44 or amp==46 or amp==48 or amp==50 or amp==52 or amp==54 or amp==56 or amp==58 or amp==60 or amp==62 or amp==64:
            corr = np.fliplr(corr)
            trim = np.fliplr(trim)

        # Create an output fits file showing the corrected frame
        filename = str(inbase) + '_fix.fits'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        if (amp == 1):
            fits.writeto(filename, corr, clobber=True)
        else:
            fits.append(filename, corr)

        # Create an output fits file showing the original frame for comparison, with pre- and post-scan trimmed
        filename = str(inbase) + '_trim.fits'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        if (amp == 1):
            fits.writeto(filename, trim, clobber=True)
        else:
            fits.append(filename, trim)

##### That's all! #####
