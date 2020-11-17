% yFit objects are things scorers operate on
classdef (Abstract) yFit
    properties (SetAccess = protected)
        yfit = [];
        Y = [];
        
        yfit_null = [];
    end
    
    % needs methods to ensure yfit and yfit_null can be fit Needs
    % reconciliation between cv.do and predictor.fit
end