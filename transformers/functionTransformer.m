% applies an arbitrary function to the data.
%
% useful for converting arbitrary canlabCore functions into transformer objects
classdef functionTransformer < Transformer
    properties (SetAccess = private)
        funhan = @(x1)(x1); % default to identity function
        
        isFitted = true;
        fitTime = 0;
    end
    
    properties (Access = ?Transformer)
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
            
            X = obj.funhan(X);
        end
    end
end
