% applies an arbitrary function to the data. Function must take fmri_data
% object as input.
classdef functionTransformer < fmriDataTransformer
    properties (SetAccess = private)
        funhan = @(x1)(x1); % default to identity function
        
        isFitted = true;
        fitTime = 0;
    end
    
    properties (Access = ?fmriDataTransformer)
        hyper_params = {};
    end
    
    methods
        function obj = functionTransformer(funhan)            
            obj.funhan = funhan;
        end
        
        function obj = fit(obj, varargin)
            t0 = tic;
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function dat = transform(obj, dat)
            assert(obj.isFitted,'Please call functionTransformer.fit() before functionTransformer.transform().');
            
            dat = obj.funhan(dat);
        end
    end
end