classdef (Abstract) modelClf < modelEstimator
    properties (SetAccess = private, Abstract)        
        decisionFcn; % used to convert raw scors to category labels
    end
    
    properties (SetAccess = protected)
        % estimated probability of positive class label during model fit,
        % used during null predictions.
        %
        % It's possible we can use a function of the offset for this, 
        % something like,
        %   yfit_null = rand(n,1) < scoreFcn(obj.offset)
        prior;
    end
    
    methods
        function yfit = predict(obj, X, varargin)            
            yfit_raw = obj.score_samples(X, varargin{:});
            yfit = obj.decisionFcn(yfit_raw);
        end
        
        % returns random class label with same probability of labels as
        % training data (it's null w.r.t. X, not Y). Note this is
        % independent of scores or decision functions.
        function yfit_null = predict_null(obj, n)
            yfit_null = zeros(n,1);
            n_pos = round(n*obj.prior);
            
            % This could also be achieved with rand(n,1) < obj.prior, but
            % class incidence wouldn't be deterministic. The way we do this
            % below, class incidence is deterministic and only the
            % particular labeling order is random.
            yfit_null(1:n_pos) = 1;
            yfit_null(n_pos+1:end) = -1;
            yfit_null = yfit_null(randperm(n));
        end
    end
end
