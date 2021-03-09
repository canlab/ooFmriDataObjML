% this is an abstract class definition for linear model objects. Linear
% model objects do not assume fmri_data objects as input, in fact they
% assume vectorized input, and consequently model weights are likewise 
% saved in vector (not fmri_data object) form. To use on fmri_data objects 
% you will need to invoke linarModelEstimators via an fmriDataEstimator 
% (*Regressor or *Clf) class instance.
%
% We have some crossed dependencies here that aren't great, but make sense
% to me so I'm keeping them. Models will be combination of (modelRegressor,
% modelClf) x (linearModelEstimator, ~).
classdef (Abstract) linearModelEstimator < baseEstimator    
    % in most cases these can just be set to empty and zero values, but we
    % define them as abstract so we can change their access properties
    % downstream for things like Dependent calls.
    % These will usually be suitable defaults
    %   B = [];
    %   offset = 0;
    properties (Abstract, SetAccess = private)
        B;
        offset;

        offset_null;
    end
    
    methods
        function yfit_raw = score_samples(obj, X, varargin)
            if isempty(obj.B)
                yfit_raw = repmat(obj.offset, size(X,1), 1);
            else
                try
                    yfit_raw = X*obj.B + obj.offset;
                catch e
                    err = struct('identfier',e.identifier,'stack',e.stack,'message',...
                        ['Problem with model fit. This may have been caused by inconsistent ',...
                        'feature extraction across folds of an optimization cv loop. Try ',...
                        'extracting features before optimization in pipeline if possible. ',...
                        'Error was: ', e.message]);
                    rethrow(err);                        
                end
            end
        end
        
        % for regressors this should be the mean observed value. For
        % classifiers this should be 0.
        function yfit_raw = score_null(obj, n)           
           yfit_raw = repmat(obj.offset_null, n, 1); 
        end
    end
end
