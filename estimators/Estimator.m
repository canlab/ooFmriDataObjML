classdef (Abstract) Estimator
    properties (Abstract, Access = ?Estimator)
        hyper_params;
    end
    
    methods (Abstract) 
        fit(obj, X, Y, varargin)
        
        % score_samples and predict are going to be the same for regression
        % (most likely, haven't thought this through fully) but for
        % classification they differ. Predict should return class labels,
        % score_samples should return continuous values, distance from
        % hyperplane, posterior probabilities, etc. score_samples will
        % usually be used for optimization (e.g. hinge loss)
        % We have a varargin because pipelines need to be able to accept
        % ('fast', true) so that fmriData transformers can evaluate 
        % efficiently during CV
        score_samples(obj, X, varargin)
        predict(obj, X, varargin)
        
        % returns a null prediction. For regressors this is just the mean
        % outcome value. For classifiers it may be more complicated.
        predict_null(obj, n)
        score_null(obj, n)

        get_params(obj)
        set_hyp(obj, hyp_name, hyp_val)
    end
end
