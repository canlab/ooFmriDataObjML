classdef (Abstract) fmriDataEstimator
    properties (Abstract, Access = ?fmriDataEstimator)
        hyper_params;
    end
    
    methods (Abstract)
        fit(obj)
	predict(obj, dat, varargin)
    end
    
    methods
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % if an estmator has hyperparameters, this sets them. 
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            obj.(hyp_name) = hyp_val;
        end       
    end
    
    methods (Access = {?crossValidator, ?fmriDataEstimator, ?fmriDataTransformer})
        function obj = compress(obj)
            return
        end
    end
end
