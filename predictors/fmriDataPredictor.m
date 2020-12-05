classdef (Abstract) fmriDataPredictor
    % we can rename fast to cv, and have it be a cross validation specific option
    % but if we do that we need to make some changes in other scripts as well that
    % currently set fast to something or other. Consider crossValPredict.m in
    % particular.
    %propeties 
    %    fast = false;
    %end
    properties (Abstract, Access = ?fmriDataPredictor)
        hyper_params;
    end
    
    methods (Abstract)
        fit(obj)
    end
    
    methods
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % if a predictor has hyperparameters, this sets them. 
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
    
    methods (Access = {?crossValidator, ?fmriDataPredictor, ?fmriDataTransformer})
        function obj = compress(obj)
            return
        end
    end
end
