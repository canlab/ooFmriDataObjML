% implements OLS regression for object oriented fmri_data predictors
%
% obj = olsRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include fixed effect intercept in GLM model. 
%                   Default: true;
classdef olsRegressor < linearModelEstimator & modelRegressor
   properties
        numcomponents = 1;
    end
    
    properties (SetAccess = protected)       
        B = [];
        offset = 0;
        offset_null = 0;
        
        intercept = true;
    end
    
    properties (Access = ?baseEstimator)
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
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
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
