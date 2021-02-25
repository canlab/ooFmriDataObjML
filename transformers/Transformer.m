classdef Transformer
    properties (SetAccess = protected)
        fitTransformTime = -1;
    end
    properties (Abstract, Access = ?Transformer)
        hyper_params;
    end
    methods (Abstract)
        fit(obj)
        transform(obj)
    end
    methods
        function [obj, dat] = fit_transform(obj, varargin)
            t0 = tic;
            obj = obj.fit(varargin{:});
            dat = obj.transform(varargin{:});
            obj.fitTransformTime = toc(t0);
        end
        
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % if a estimator has hyperparameters, this sets them. 
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            obj.(hyp_name) = hyp_val;
        end
    end
end
