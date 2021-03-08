classdef (Abstract) baseEstimator < handle & dynamicprops & matlab.mixin.Copyable
    properties (Abstract, Access = ?Estimator)
        hyper_params;
    end
    
    methods (Abstract) 
        fit(obj, X, Y, varargin)
        
        % score_samples and predict are going to be the same for regression
        % (most likely, haven't thought this through fully) but for
        % classification they differ. Predict should return class labels,
        % score_samples should return continuous values, distance from
        % hyperplane, posterior probabilities, etc. score_samples will
        % usually be used for optimization (e.g. hinge loss)
        % We have a varargin because pipelines need to be able to accept
        % ('fast', true) so that fmriData transformers can evaluate 
        % efficiently during CV
        score_samples(obj, X, varargin)
        predict(obj, X, varargin)
        
        % returns a null prediction. For regressors this is just the mean
        % outcome value. For classifiers it may be more complicated.
        predict_null(obj, n)
        score_null(obj, n)
    end
    
    methods
        function params = get_params(obj)
            params = obj.hyper_params;
        end

        % if an estimator has hyperparameters, this sets them.
        function set_params(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));

            obj.(hyp_name) = hyp_val;
        end
        
        % adds hyperparameters dynamically. Useful for multiclass
        % classifiers
        function addDynProp(obj, hyp_name)
            addprop(obj, hyp_name);
        end
        
        function estimator = getBaseEstimator(obj)
            estimator = obj;
            
            fnames = fieldnames(obj);
            for i = 1:length(fnames)
                if isa(obj.(fnames{i}),'baseEstimator')
                    estimator = getBaseEstimator(obj.(fnames{i}));
                    break;
                end
            end
        end
    end
    
    methods (Access = protected)
        function obj = copyElement(obj)
            obj = copyElement@matlab.mixin.Copyable(obj);
            
            fnames = fieldnames(obj);
            for i = 1:length(fnames)
                if isa(obj.(fnames{i}), 'matlab.mixin.Copyable')
                    obj.(fnames{i}) = copy(obj.(fnames{i}));
                elseif isa(obj.(fnames{i}), 'handle') % implicitly: & ~isa(obj.(fnames{i}), 'matlab.mixin.Copyable')
                    % the issue here is that fuction handles that are
                    % copied can contain references to the object they
                    % belong to, but these references will continue to
                    % point to the original object, and not the copy
                    % becaues matlab cannot parse these function handles
                    % appropriately.
                    warning('%s.%s is a handle but not copyable. This can lead to unepected behavior and is not ideal', class(obj), fnames{i});
                end
            end
        end
    end
end
