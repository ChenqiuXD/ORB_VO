##OverView
This is an implementation of the journal "Real-Time Stereo Visual Odometry for Autonomous Ground Vehicles" by Andrew Howard

##Some changes
In A2 step, the author use a rectangular as descriptors, yet we use the opencv built-in ORB method which use BRIEF descriptor
In A2 and A3 steps, the author use SAD to calculate a score matrix to match the features. We use the opencv function BFMatch.match() instead.

##Unfinished part
A5 step which calculate the movement by matches remains unfinished
A6,7 steps which evaluate the ORB VO result is unfinised
The output of trajectory, which is required to be a .txt file is not finished yet.