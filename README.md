# skymapper
Standalone scripts and examples for the correction of bias noise and fringing in SkyMapper images

bias/
Scripts to remove row-based electronic noise from SkyMapper images with a PCA-based algorithm
run_bias_correction.pl is a helper script that prepares the data and runs the two python scripts in order for a given night
bias_magic.py learns the nightly PCs describing the row-oriented noise patterns in SkyMapper images
science_magic.py uses the nightly PCs to correct bias noise in all SkyMapper science images from that night
bias_correct.sxt, bias_correct.conv, bias_correct.param contain parameters controlling Source Extractor
three example outcomes (before & after) are provided - basic, crowded, nebulosity

fringe/


Some brief notes on SkyMapper images:
The focal plane of the SkyMapper camera is tiled with 32 CCDs containing 2048x4096 pixels each (268.4 Mpix total).
Each CCD is read out through 2 amplifiers, with the split at column 1024. Hence a given image has 64 extensions ("frames").
Since the 2 amps read in opposite directions, half of the frames need their x-coordinates flipped before analysis.
During readout, an extra 50 pix before and after the data are also recorded ("pre-scan" and "post-scan").
The idea behind this is to allow the current base level of the electronics to be corrected.
This is typically set to be significantly above zero to eliminate negative pixel values.
At the beginning and end of a given night, a set of ~ten "bias" images are saved.
These have zero exposure time (i.e., are readout only), recording the row-oriented noise patterns for that night.
Images taken using very red filters exhibit a structured "fringing" pattern at a level up to ~10% of background (sky).
These are due to thin-film interference effects in the detector when illuminated with long-wavelength photons.
Here the illumination is provided by groups of night-sky emission lines, so the fringe intensity varies with time.
SkyMapper images are stored in the astronomical standard fits format, see https://en.wikipedia.org/wiki/FITS

