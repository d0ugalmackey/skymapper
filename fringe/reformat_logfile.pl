#!/usr/local/bin/perl

# Helper script to reformat the logfile of information created by the preprocessing routine preprocess_images.py
# Merge this with some information from the original image list made by create_image_list.pl
# This new data file will be ingested by the fringe creation routine and used to select the best images

if ($#ARGV != 2) {die "\nUsage: process_log.pl <input logfile> <original image list> <outfile>\n\n";}

($infile) = $ARGV[0];
($basefile) = $ARGV[1];
($outfile) = $ARGV[2];

open (LOG, "$infile");
open (LIST, "$basefile") || die "The original list of input images is also required!\n";
open (OUT, ">$outfile");

$_ = <LOG>;
$_ = <LOG>;
$_ = <LOG>;
$_ = <LOG>;

$ntry = 0;

# Increment through all the images in the original list and match with those in the log file
# The order should be the same, but double check to be conservative
while (<LIST>) {
  # Get the date and exposure time from the original image list
  chomp; @_ = split;
  $date = @_[0];
  $image = @_[1];
  $exp = @_[4];

  # Get all the pixel statistics from the preprocessing log
  $_ = <LOG>;
  chomp; @_ = split;
  $image2 = @_[7];
  $mean = @_[9];
  $stdev = @_[10];
  $mask = @_[11];
  $sat = @_[12];
  $backgr = @_[13];
  
  $ntry ++;

  # Check that things didn't get out of whack
  if ($image ne $image2) {die "Image mismatch!  $image  $image2\n";}

  # Print merged info to the new file for ingest
  print OUT "$image\t$date\t$exp\t$mean\t$stdev\t$mask\t$sat\t$backgr\n";
}

close (LOG);
close (OUT);
close (LIST);

print "$ntry images processed\n";


  
