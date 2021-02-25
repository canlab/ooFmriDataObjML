classdef (Abstract) Estimator
    properties (Abstract, Access = ?Estimator)
        hyper_params;
    end
    
    methods (Abstract) 
        fit(obj, X, Y)
        predict(obj, X)
              
        get_params(obj)
        set_hyp(obj, hyp_name, hyp_val)
    end
end