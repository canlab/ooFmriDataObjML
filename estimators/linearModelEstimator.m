% this is an abstract class definition for linear model objects. Linear
% model objects do not assume fmri_data objects as input, in fact they
% assume vectorized input, and consequently model weights are likewise 
% saved in vector (not fmri_data object) form. To use on fmri_data objects 
% you will need to invoke linarModelEstimators via an fmriDataEstimator 
% (*Regressor or *Clf) class instance.
classdef (Abstract) linearModelEstimator < modelEstimator    
    properties (SetAccess = ?linearModelEstimator)        
        B = [];
        offset = 0;
    end
end