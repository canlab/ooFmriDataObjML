classdef (Abstract) modelClf < baseEstimator
    
    properties (SetAccess = protected)
        % estimated probability of positive class label during model fit,
        % used during null predictions.
        %
        % It's possible we can use a function of the offset for this, 
        % something like,
        %   yfit_null = rand(n,1) < scoreFcn(obj.offset)
        prior;
    end
    
    properties (Abstract, SetAccess = private)
        classLabels;
    end
    
    properties (SetAccess = private)
        offset_null = 0;
    end
    
    methods (Abstract)
        % a decision function that converts scores into class labels.
        % Typically scores > 0 might do, but for SVM we need -1/1 labels
        % for instance, while for multiclass classification we need
        % something more complex that combines the conclusions of all
        % constituent binary classifiers. We define this as abstract to
        % force the user to implement something intelligent.
        decisionFcn(obj, scores);
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
            if length(obj.classLabels) == 2
                prior = [1-obj.prior; obj.prior];
            else
                prior = obj.prior;
            end
            n_pos = floor(n*prior);
            
            yfit_null = [];
            for i = 1:length(obj.classLabels)
                yfit_null = [yfit_null; repmat(obj.classLabels(i), n_pos(i), 1)];
            end
            if length(yfit_null) < n % deal with rounding errors
                missing = n - length(yfit_null);
                
                % get random entries to padd the results
                u = rand(missing,1) ;
                % convert ressultant random [0,1] values 
                % into labels using intervals on the uniform 
                % distribution to model priors likelihoods for each 
                % label
                r = u < cumsum(prior(:))';
                label = [~any(diff(r, [], 2),2), diff(r, [], 2)];
                
                % convert indexed matrix into corresponding labels
                [a,b] = find(label);
                [~,I] = sort(a);
                label_idx = b(I);
                
                yfit_null = [yfit_null; obj.classLabels(label_idx)];
            end
                
            % shuffle the labels    
            yfit_null = yfit_null(randperm(n));
        end
    end
end
