% this is an abstract class definition for linear model objects. Linear
% model objects do not assume fmri_data objects as input, in fact they
% assume vectorized input, and consequently model weights are likewise 
% saved in vector (not fmri_data object) form. To use on fmri_data objects 
% you will need to invoke linarModelEstimators via an fmriDataEstimator 
% (*Regressor or *Clf) class instance.
%
% We have some crossed dependencies here that aren't great, but make sense
% to me so I'm keeping them. Models will be combination of (modelRegressor,
% modelClf) x (linearModelEstimator, nonlinearModelEstimator).
classdef (Abstract) linearModelEstimator < modelEstimator    
    properties (SetAccess = ?linearModelEstimator)        
        B = [];
        offset = 0;
    end
    
    methods
        function yfit_raw = score_samples(obj, X, varargin)
            yfit_raw = X*obj.B(:) + obj.offset;
            
            yfit_raw = yfit_raw(:);
        end
        
        function yfit_null = score_null(obj)
            yfit_null = obj.offset;
        end
    end
end
