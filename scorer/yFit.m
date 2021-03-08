% yFit objects are things scorers operate on. For classifier scorers may
% either operate on labels (yfit) or on unlabeled outcome values
% (yfit_raw), depending on the scorer. For instance hinge loss operates on
% raw scores but f1 is computed based on labels. Regression scorers 
% operate on yfit, but yfit and yfit_raw should be the same for regression.
classdef (Abstract) yFit < handle & matlab.mixin.Copyable
    properties (SetAccess = protected)
        yfit = [];      % predicted scores or category labels
        yfit_raw = [];  % same as yfit for regression, but raw scores for categorical outcomes
        yfit_null = [];
        
        Y = [];
    end
    
    properties (Abstract)
        % used in multiclass classification. Order must match order of columns returned by estimator.score_samples()
        classLabels; 
    end
    
    % needs methods to ensure yfit and yfit_null can be fit Needs
    % reconciliation between cv.do and predictor.fit
end