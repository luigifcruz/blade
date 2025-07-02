# Spectral Kurtosis

## References

https://www.worldscientific.com/doi/10.1142/S225117171940004X (1)

## Overview

The goal of this kernel is to, given a block of data from the ATA, replace certain RFI-polluted channels with values that are more acceptable and conducive to productive data analysis.

We will remove channels if they are not Gaussian enough -- this is evaluated by calculating the spectral kurtosis of each channel, as shown in formula (2) in reference [1]. The channel is removed if the spectral kurtosis value falls outside of the bounds that we get by running the `sklimit` command for the appropriate number of samples (M) and strength of threshold (# of stddevs).

We determined empirically that a kurtosis block size (M, in the formula) of 256 is ideal, along with a stddev of 5 for the kurtosis bounds.

The kurtosis kernel goes through each sub-channel and pol, calculating the spectral kurtosis (SK) values for each of the 256-sample long channels. 


