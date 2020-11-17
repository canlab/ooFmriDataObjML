classdef (Abstract) fmriDataPredictor
    methods (Abstract)
        fit(obj)
        
        % should return an objects hyperparameter variable names
        get_params(obj)
    end
    
    methods
        
        % if a predictor has hyperparameters, this sets them. They should
        % be entered in alphabetic order with respect to their property
        % names
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            obj.(hyp_name) = hyp_val;
        end
        
        function yfit = predict(obj, dat)
            assert(obj.isFitted,sprintf('Please call %s.fit() before %s.predict().\n',class(obj)));
            yfit = apply_mask(dat, obj.weights, 'pattern_expression', 'dotproduct', 'none') + obj.offset;
            yfit = yfit(:);
        end
    end
end
