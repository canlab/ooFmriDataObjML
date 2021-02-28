% this is an abstract class definition for model objects. 
% model objects do not assume fmri_data objects as input, in fact they
% assume vectorized input. To apply these to fmri_data objects you
% will need to invoke them through an fmriDataEstimator object
% modelEstimators contain the internal logic of any particular 
% MVPA algorithm, separate from any fmri_data specific operations.
% They can be linear or nonlinear, and separate abstract classes
% exist for each.
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
