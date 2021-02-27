% implements OLS regression for object oriented fmri_data predictors
%
% obj = olsRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include fixed effect intercept in GLM model. 
%                   Default: true;
 classdef olsRegressor < linearModelRegressor
   properties
        numcomponents = 1;
    end
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
        
        intercept = true;
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'intercept'};
    end
    
    methods
        function obj = olsRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'intercept'
                            obj.intercept = varargin{i+1};
                    end
                end
            end
        end
        
        function obj = fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            
            if obj.intercept
                X = [ones(length(Y),1), X];
            end
            b = regress(Y,X);
            
            obj.B = b(2:end);
            obj.offset = b(1);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end