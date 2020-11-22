# ooFmriDatML
Object oriented ML framework for fmri_data objects

Inpsired by the design of scikit-learn. Designed to balance interoperability
with the sckit-learn syntax (for ease of use and adoption), but operates 
exclusively on canlabCore fmri_data objects. More geared towards preprocessing
than feature construction.

### Dependencies
* https://github.com/canlab/ComBatHarmonization
* https://github.com/canlab/CanlabCore
* https://github.com/canlab/canlab_single_trials (recommended)

### To Do
scikit learn is powerful in part because it allows for very flexible feature
construction. This library isn't very good at this because of the way it
incorporates fmri_data objects into its input. For instance, suppose you wanted
to develop a classifier that makes predictions based on net contrast in a 
set of a priori regions of interest. Suppose you select p regions, so now your 
feature vector is 1 x p. This feature vector no longer has any meaningful 
representation as an fmri_data object and cannot be supplied as input for any
of the fmriDataPredictor objetcts in this repo. Nevertheless, some kind of
fmri_data representation is needed because often times implementing new data
requires some kind of metadata awareness by the algorithm. For instance, when
applying a new regressor, we must make sure that the input feature space matches
the model feature space, which is anatomicaly defined. We also need batch 
awareness for any algorithms (e.g. cross validation, combat, mlpcr, etc).

One way around this might be to create a class that instantiates a particular
fmri_data object's space, and can project any new fmri_data objects into the 
original fmri_data objects space. We can then augment the pipeline method to
accept this class as its first transformer. This doesn't solve the problem of
how we package metadata with our object (which may often be needed). We can 
solve this by modifying all do, transform and fit functions to also take a 
metadata field as input. This field can be empty, but it's also available for 
anonyous functions various classes might call upon. Given a feature matrix X 
that is n x p, the metadata field would be constrained to be n x m.


