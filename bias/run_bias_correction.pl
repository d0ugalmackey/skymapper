#!/usr/local/bin/perl

## Helper script to automate running of PCA-based noise removal
## PCs are computed from the bias (zero-exposure) frames for the night using bias_magic.py
## These are then applied to a list of science frames observed on the same night using science_magic.py
## Assume the data are in pwd and the required software is in ../code

# First retrieve the necessary bits and pieces of code
# This includes the two main python scripts, and three parameter files for Source Extractor 
system ("cp ../code/bias_magic.py .");
system ("cp ../code/science_magic.py .");
system ("cp ../code/bias_correct.sxt .");
system ("cp ../code/bias_correct.conv .");
system ("cp ../code/bias_correct.param .");

# Now create a text file containing the list of bias frames in pwd
# These are named brains53_187*
# Note that "brains" indicates the SDP has already applied the removal of sinusoidal interference pattern
# Code 187 indicates bias (zero-exposure) frames taken at the start and/or end of the night
system ("ls brains53_187*.fits > biaslist.dat");

# Now create a text file containing the list of science frames in pwd
# These are named brains53_* but cannot have code 187 (which indicates biases)
# First create a temporary file listing all images
unlink "temp.dat" if -e "temp.dat";
system ("ls brains53_*.fits > temp.dat");

# Now run through this list and keep only lines that are not biases
open (TEMP, "temp.dat");
open (SCI, ">sciencelist.dat");
while (<TEMP>) {
  # Use regex to search for code 187
  chomp;
  ($base) = /(\S+)\.fits/;
  $_ = $base;
  ($code) = /\S+\_(\d\d\d)\d+\_\S+/;
  if ($code != 187) {
    print SCI "$base\n";
  }
}
close (TEMP);
close (SCI);
unlink "temp.dat";

# Now run the python code -- bias_magic.py followed by science_magic.py
# Note the dependencies listed in the two scripts
system ("python bias_magic.py");
system ("python science_magic.py");

## That's it! ##

