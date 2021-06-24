# ooFmriDatML
Object oriented ML framework for fmri_data objects

Inpsired by the design of scikit-learn. Designed to balance interoperability
with the sckit-learn syntax (for ease of use and adoption), but designed to
work with block dependencies like subjects with multiple observations. 

### Dependencies
* https://github.com/canlab/ComBatHarmonization
* https://github.com/canlab/CanlabCore
* https://github.com/canlab/canlab_single_trials (recommended)

### To Do
scikit learn is powerful in part because it allows for very flexible feature
construction. This is problematic with fmri_data objects because features must 
exist in some kind of MNI space, but not all features you might want to use
do exist in this space in a natural way. Consider if all you're interested in 
are coarse signals, like net activity in a subset of regions of interest. There's 
no good way to represent these features as an fmri_data object, it's much more 
sensible taken out of brain space. However, you must have some kind of MNI space 
awareness for testing new input which may not be in the same space.

There are several options for working around this. The one used here is to define a 
transformer that converts from fmri_data objects to features, and which when
'fit', saves the reference images space as a property, so that when substequent
transformations are applied, data can be projected into this space and then
have its \*.dat field returned. The limitation is that we need a way to 
package metadata downstream, specifically block id metadata for things like
group k-fold CV and block aware modeling algorithms (e.g. concensus PCA, mixed
effects models, etc). The way we solve that is to give features objects a metadata
property. The features object is in fact what's know as an extended double, in other
words just a vector of doubles but extended in the sense that it has properties. 
Metadata must implement subscripting and concatenation operators, so for instance
other vectors or tables implement are suitable, but they can also be implemented for
abitrary customized classes as indicated here:
https://www.mathworks.com/help/matlab/matlab_oop/implementing-operators-for-your-class.html
which allows for pretty flexible metadata use.

The neat thing about this approach is that it means all the base ML algorithms of 
this library can actually be used on non-fmri_data objects too, you just load them up
in a features object if they need metadata for block awareness, or supply them 
directly to the algorithms as vectors/matrices.

### Developer notes
This library makes extensive use of handles. Most objects here inheret the 
handle class in fact, and this leads to certain important notes regarding
handle classes.
1) handles are passed by reference, not by value. To pass by value, invoke
obj.copy(), and if updating handle classes make sure that obj.copy() is updated
to accomodate your updates appropriately
2) handle object modifications do not automatically propogate out of parallel
workers. To propogate back out explicitly assign the modified handles to output
variables. See here for details: 
https://www.mathworks.com/help/parallel-computing/objects-and-handles-in-parfor-loops.html
