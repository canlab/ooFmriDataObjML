% yFit objects are things scorers operate on
classdef (Abstract) yFit
    properties (SetAccess = protected)
        yfit = [];      % predicted scores or category labels
        yfit_raw = [];  % same as yfit for regression, but raw scores for categorical outcomes
        yfit_null = [];
        
        Y = [];        
    end
    
    % needs methods to ensure yfit and yfit_null can be fit Needs
    % reconciliation between cv.do and predictor.fit
end