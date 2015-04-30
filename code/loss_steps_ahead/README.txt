NOTES:

### dpp  
Folder with code to sample from dpps. Based on the code of Alex Kulesza. Contains two new functions:
- sample_conditional_dpp: to samples from a conditional dpp given a set.
- sample_dual_conditional_dpp: the same but based on a kernel truncation that makes it faster

### emin_epmgp 

Folder based on the code of John P Cunningham for EP with Gaussian densities. Contains a new function:
- emin_epmgp that calculates the expectation of the minium of a gaussian vector (and a constant).

### gpml
Just the gpml library, used for the tests

Appart from this these folders:
- loss_msahead: computes the loss function for multiple steps ahead decision
- test_sample_conditional.m, test_emin_epmgp.m are test to sample form the dpp and compute the integral
- test_loss_msahead.m is a test to visualize the computed loss in a toy 2D example (takes a while). The pictures are the original function, the model and the computed loss.






 










