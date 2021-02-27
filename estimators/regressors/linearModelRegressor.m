% Regressor based linear models assume a prediction formula, which is
% implemented here, but the abstract class is also a useful identifier that
% can be used by fmriDataEstimators to correctly implement regression 
% models on fmri_data objects.
classdef (Abstract) linearModelRegressor < linearModelEstimator    
    methods
        function yfit = predict(obj, X, varargin)            
            yfit = X*obj.B(:) + obj.offset;
            
            yfit = yfit(:);
        end
    end
end