% applies an arbitrary function to the data.
%
% useful for converting arbitrary canlabCore functions into transformer objects
classdef functionTransformer < baseTransformer
    properties (SetAccess = private)
        funhan = @(x1)(x1); % default to identity function
        
        isFitted = true;
        fitTime = 0;
    end
    
    properties (Access = ?baseTransformer)
        hyper_params = {};
    end
    
    methods
        function obj = functionTransformer(funhan)            
            obj.funhan = funhan;
        end
        
        function fit(obj, varargin)
            t0 = tic;
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function X = transform(obj, X)
            assert(obj.isFitted,'Please call functionTransformer.fit() before functionTransformer.transform().');
            
	    if isa(X,'features')
	        X = features(obj.funhan(X), X)
	    else
                X = obj.funhan(X);
            end
        end
    end
end
