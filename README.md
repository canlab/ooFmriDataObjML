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
incorporates fmri_data objects into its input. A features must exist in some
kind of MNI space as a result, but not all features you might want to use
do exist in this space. Consider if all you're interested in is coarse signals,
like net activity in a subset of regions of interest. There's no good way to
represent these features as an fmri_data object, it's much more sensible 
taken out of brain space. However, you must have some kind of MNI space 
awareness for testing new input which may not be in the same space.

There are several options for working around this. One is to define a 
transformer that converts from fmri_data objects to features, and which when
'fit', saves the reference images space as a property, so that when substequent
transformations are applied, data can be projected into this space and then
have its \*.dat field returned. The limitation is that we need a way to 
package metadata downstream, specifically block id metadata for things like
group k-fold CV and block aware modeling algorithms (e.g. concensus PCA, mixed
effects models, etc). Another fix to this might be for all fit, do and transform
functions to take a third (optionally null) argument that provides metadata.
The metadata would only need to implement subscripting and concatenation 
operators, which tables implement, but which can also be implemented for
abitrary customized classes as indicated here:
https://www.mathworks.com/help/matlab/matlab_oop/implementing-operators-for-your-class.html
which allows for flexible metadata use.

Here's a new better idea,
Overload indexing of fmri_data objects (cat is already implemented). Add a 'size' and
'length' method to the class in canlabCore (should be an improvement without any
negative side effects). Convert fmriDataPredictors to invoke a get_X() anonymous 
function on input data to extract features. This can default to @(x1)(x1.dat') by
default, but can be modified by class invocation to look elsewhere. You can't use
fmri_data anywhere in the input to the predictors for this to work, it can't assume
the existence of fmri_data specific metadata, which will make image fitting
difficult. Instead create a transformer class that takes fmri_data objects and
converts them to simple X block aware structures.
struct('X', dat.dat, 'block_id', dat.metadata_table.block_id);
