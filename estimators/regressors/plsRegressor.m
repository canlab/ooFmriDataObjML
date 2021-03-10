classdef plsRegressor < linearModelEstimator & modelRegressor
    properties
        numcomponents = 1;
    end
    
    properties (SetAccess = protected)                
        isFitted = false;
        fitTime = -1;
        
        B = [];
        offset = 0;
        offset_null = 0;
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'numcomponents'};
    end
    
    methods
        function obj = plsRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
                    end
                end
            end
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
            if ~isempty(obj.numcomponents)
                [~,~,~,~,b] = plsregress(X, Y, obj.numcomponents);
            else
                [~,~,~,~,b] = plsregress(X, Y);
            end
            obj.B = b(2:end);
            obj.offset = b(1);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end

