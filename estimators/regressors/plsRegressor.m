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
            
            [n,dx] = size(X);
            ny = size(Y,1);
            assert(ny == n, 'length(Y) ~= size(X, 1)');

            % Return at most maxncomp PLS components
            maxncomp = min(n-1,dx);
            if maxncomp < obj.numcomponents
                warning('%d components requested but only %d possible. Using %d', ...
                    obj.numcomponents, maxncomp, maxncomp);
                obj.numcomponents = maxncomp;
            end
            
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

