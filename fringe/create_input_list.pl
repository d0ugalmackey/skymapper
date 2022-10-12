#!/usr/local/bin/perl

# Helper script to create a list of input images for fringe PC creation given a source directory and date range
# Also specify the amp and filter of interest (so this can be looped over all amps and relevant filters if desired)
# The source directory is the location of partially-processed data from the SDP
# Dates are specified as MJD (modified julian date) -- for example, 12 Oct 2022 is night 59864
# Amp is an integer from 1 to 64, and filter should be i or z (since fringes only appear for these very red settings)
# Note that the source dir contains images taken with all filters, so we need to explicitly check for i,z

if ($#ARGV != 4) {die "\nUsage: create_input_list.pl <base directory> <start date> <end date> <filter> <amp>\n\n";}

($basedir) = $ARGV[0];
($mjd1) = $ARGV[1];
($mjd2) = $ARGV[2];
($filt) = $ARGV[3];
($amp) = $ARGV[4];

# Store the list of images to use in input_images.dat
open (OUT, ">input_list.dat");

# Increment through the specified date range
for ($mjd = $mjd1; $mjd <= $mjd2; $mjd ++) {
  # Specify the location of the reduced science images for this night and amp 
  $source = $basedir . "\/" . $mjd . "\/" . $amp . "\/";

  # Get rid of local temp files, if they still exist from previous runs
  unlink "filters1.tmp" if -e "filters1.tmp";
  unlink "filters2.tmp" if -e "filters2.tmp";
  unlink "exposures.tmp" if -e "exposures.tmp";

  # Get information from the fits headers for all science images in the specified location
  # Do two filter checks since the format of the fits header can vary with date and SDP version
  system ("gethead -p FILTNAME $source\/\*red.fits \> filters1.tmp");      # Lists the filter used for each image
  system ("gethead -p FILTER $source\/\*red.fits \> filters2.tmp");
  system ("gethead -p EXPTIME $source\/\*red.fits \> exposures.tmp");      # Lists the exposure time of each image

  # Now go through and select only images of interest (correct filters and long exposure times)
  open (FILT1, "filters1.tmp");
  open (FILT2, "filters2.tmp");
  open (EXP, "exposures.tmp");
  $ntry = 0; $nfilt = 0; $ngood = 0;
  
  while (<FILT1>) {
    # Get the filter, and the name of the image, from the first temp file
    chomp; @_ = split;
    $_ = @_[0]; $filter1 = @_[1];
    ($image) = /\S+\/(\S+\.fits)/;
    
    # The filter might actually be indexed with a different header keyword, so keep that info too
    $_ = <FILT2>;
    chomp; @_ = split;
    $filter2 = @_[1];

    # And get the exposure time
    $_ = <EXP>;
    chomp; @_ = split;
    $exptime = @_[1];
    $ntry ++;

    # Only keep images that were taken with the filter of interest
    # Also exclude images with exposure time < 60s.
    # We need the fringe signal to be substantially higher than the pixel-to-pixel noise of the background
    # This is only satisfied in images with decently long exposure time (empirically >60s seems ok)
    if (($filter1 eq $filt) || ($filter2 eq $filt)) {
      $nfilt ++;
      if ($exptime > 60) {
	# Write useful images to the output list
	print OUT "$mjd   $image   $filter1   $filter2   $exptime\n";
	$ngood ++;
      }
    }
  }
  close (FILT1);
  close (FILT2);
  close (EXP);

  # Clean up the temp files
  unlink "filters1.tmp" if -e "filters1.tmp";
  unlink "filters2.tmp" if -e "filters2.tmp";
  unlink "exposures.tmp" if -e "exposures.tmp";

  # Log the success rate for this night
  print "$mjd\t$ntry\t$nfilt\t$ngood\n";
}

close (OUT);
