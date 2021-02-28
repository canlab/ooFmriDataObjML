% Regressor based models have the same prediction as scores, but it's also
% useful to have a regressor identifier so that downstream objects know
% whether they're dealing with a regressor or a classifier.
%
% We have some crossed dependencies here that aren't great, but make sense
% to me so I'm keeping them. Models will be combination of (modelRegressor,
% modelClf) x (linearModelEstimator, nonlinearModelEstimator).
classdef (Abstract) modelRegressor < modelEstimator    
    methods
        function yfit = predict(obj, X, varargin)
            yfit = obj.score_samples(X, varargin{:});
        end
        
        function yfit_null = predict_null(obj)
            yfit_null = obj.score_null(obj);
        end
    end
end
