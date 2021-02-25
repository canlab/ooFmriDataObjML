% this is an abstract class definition for linear model objects. Linear
% model objects do not assume fmri_data objects as input, in fact they
% assume vectorized input, and consequently model weights are likewise 
% saved in vector (not fmri_data object) form. To use on fmri_data objects 
% you will need to invoke linarModelEstimators via an fmriDataEstimator 
% (*Regressor or *Clf) class instance.
classdef (Abstract) modelEstimator < Estimator 
    methods         
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % if an estimator has hyperparameters, this sets them. 
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            obj.(hyp_name) = hyp_val;
        end       
    end
end