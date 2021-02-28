classdef (Abstract) linearModelClf < modelEstimator
    properties (Abstract)
        decisionFcn; % used to convert raw scors to category labels
    end
    
    methods (Abstract)
        predict_null(obj, X, varargin);
    end
    
    methods
        function yfit = predict(obj, X, varargin)            
            yfit_raw = obj.score_samples(X, varargin{:});
            yfit = obj.decisionFcn(yfit_raw);
        end
    end
end