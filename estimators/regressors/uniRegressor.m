% uniRegressor a linearModelEstimator that fits separate univariate models 
%   for each input feature and uses model averaging (equal weighting across
%   univariate models) to obtain a final prediction. Implementation differs
%   from fmri_data/predict.cv_univregress but result should be the same.
%
% estimator = uniRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include intercept in univariate model. 
%                   Default: true;
classdef uniRegressor < linearModelEstimator & modelRegressor
   properties
        numcomponents = 1;
    end
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
        
        B = [];
        offset = 0;
        
        intercept = true;
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'intercept'};
    end
    
    methods
        function obj = uniRegressor(varargin)
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
                b = zeros(size(X,2),2);
            else
                b = zeros(size(X,2),1);
            end
            
            for i = 1:size(X,2)
                % X is brain, Y is outcome, for each voxel
                xx = [ones(size(Y)) X(:, i)];

                %b = pinv(X) * Y
                b(i,:) = xx \ Y;   % should be same, but faster
            end
            
            obj.B = b(:,2)/size(b,1);
            obj.offset = mean(b(:,1));
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end
