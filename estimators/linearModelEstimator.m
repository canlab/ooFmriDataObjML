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
    % in most cases these can just be set to empty and zero values, but we
    % define them as abstract so we can change their access properties
    % downstream for things like Dependent calls.
    % These will usually be suitable defaults
    %   B = [];
    %   offset = 0;
    properties (Abstract, SetAccess = private)
        B;
        offset;
    end
    
    methods
        function yfit_raw = score_samples(obj, X, varargin)
            yfit_raw = X*obj.B(:) + obj.offset;
            
            yfit_raw = yfit_raw(:);
        end
        
        % not sure if this is valid for classifiers with scoreFcns. The raw
        % null score for a linearSvmClf with a nonlinear scoreFcn may be
        % obj.scoreFcn(obj.offset). Need to check this, and if so move
        % score_raw into modelRegressor, and leave it abstract in modelClf,
        % implementing it on a case by case basis in modelClf subclases.
        % Right now I'm just overloading this in modelClf instances that
        % use scoreFcn's and throwing a warning if they're non-trivial
        function yfit_raw = score_null(obj, n)           
           yfit_raw = obj.offset*ones(n,1); 
        end
    end
end
