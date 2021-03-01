classdef (Abstract) modelClf < modelEstimator
    properties (Abstract)        
        decisionFcn; % used to convert raw scors to category labels
    end
    
    methods
        function yfit = predict(obj, X, varargin)            
            yfit_raw = obj.score_samples(X, varargin{:});
            yfit = obj.decisionFcn(yfit_raw);
        end

        function yfit_null = predict_null(obj, varargin)
            yfit_raw = obj.score_null(obj);
            yfit_null = obj.decisionFcn(yfit_raw);
        end
    end
end
