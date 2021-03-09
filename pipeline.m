classdef pipeline < baseEstimator & baseTransformer
    properties 
        verbose = false;
    end
    properties (SetAccess = private)
        % the following are private because names and associated objects need to be modified
	% simultaneously. We have methods for this below
        transformers = {};
        transformer_names = {};
        estimator = {};
        estimator_name = [];

        isFitted = false;
        
        fitTime = -1
    end
    
    properties (Access = {?baseTransformer, ?baseEstimator})
        hyper_params = {};
    end
    
    methods
        function obj = pipeline(steps, varargin)
            for i = 1:length(steps) - 1
                assert(isa(steps{i}{2}, 'baseTransformer'), ...
                    sprintf('All but last step must be transformers, but element %d is type %s',i,class(steps{i}{2})));
                
                obj.transformer_names{end+1} = steps{i}{1};
                obj.transformers{end+1} = copy(steps{i}{2});
            end
            if isa(steps{end}{2},'baseEstimator')
                obj.estimator_name = steps{end}{1};
                obj.estimator = copy(steps{end}{2});
            elseif isa(steps{end}{2},'baseTransformer')
                obj.transformer_names{end+1} = steps{end}{1};
                obj.transformers{end+1} = copy(steps{end}{2});
            else
                error('Last step must be a transformer or a estimator');
            end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'verbose'
                            obj.verbose = varargin{i+1};
                    end
                end
            end
            
            % set hyperparameters to match whatever the constituent step
            % hyperparameters are
            params = {};
            for i = 1:length(obj.transformers)
                these_params = obj.transformers{i}.get_params();
                params = [params, cellfun(@(x1)([obj.transformer_names{i}, '__', x1]), ...
                   these_params, 'UniformOutput', false)];
            end
            if ~isempty(obj.estimator)
                warning('off','bayesOptCV:get_params')
                these_params = obj.estimator.get_params();
                warning('on','bayesOptCV:get_params');
                
                params = [params, cellfun(@(x1)([obj.estimator_name, '__', x1]), these_params, 'UniformOutput', false)];
            end
            
            obj.hyper_params = params;
        end
        
        % fit all transformers and any estimators
        function fit(obj, dat, Y)
            t0 = tic;
            for i = 1:length(obj.transformers)
                % output from one transformer is input to the next
                if obj.verbose, fprintf('Fitting %s\n', obj.transformer_names{i}); end
                dat = obj.transformers{i}.fit_transform(dat);            
            end
            if ~isempty(obj.estimator)
                if obj.verbose, fprintf('Fitting %s\n', obj.estimator_name); end
                
                obj.estimator.fit(dat, Y);
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        % apply all transforms
        function dat = transform(obj, dat, varargin)
            for i = 1:length(obj.transformers)
                if obj.verbose, fprintf('Applying %s\n', obj.transformer_names{i}); end
                % output from one transformer is input to the next
                dat = obj.transformers{i}.transform(dat);
            end
        end
        
        % apply all transforms and predict
        function yfit_raw = score_samples(obj, dat, varargin)
            assert(~isempty(obj.estimator), ...
                'This pipeline does not terminate in a estimator. Try pipeline.transform() instead');
            
            dat = obj.transform(dat, varargin{:});
            if ~isempty(obj.estimator)
                if obj.verbose, fprintf('Applying %s\n', obj.estimator_name); end
                yfit_raw = obj.estimator.score_samples(dat);
            end
        end
        
        % apply all transforms and predict
        function yfit = predict(obj, dat, varargin)
            assert(~isempty(obj.estimator), ...
                'This pipeline does not terminate in a estimator. Try pipeline.transform() instead');
            
            dat = obj.transform(dat, varargin{:});
            if ~isempty(obj.estimator)
                if obj.verbose, fprintf('Applying %s\n', obj.estimator_name); end
                yfit = obj.estimator.predict(dat);
            end
        end    
        
        
        % apply all transforms and predict
        function yfit_null = score_null(obj, varargin)
            assert(~isempty(obj.estimator), ...
                'This pipeline does not terminate in a estimator. Try pipeline.transform() instead');
            
            yfit_null = obj.estimator.score_null(varargin{:});
        end    
        
        % apply all transforms and predict
        function yfit_null = predict_null(obj, varargin)
            assert(~isempty(obj.estimator), ...
                'This pipeline does not terminate in a estimator. Try pipeline.transform() instead');
            
            yfit_null = obj.estimator.predict_null(varargin{:});
        end    
        
        
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % finds object to modify and calls its obj.set_params(passThrough,
        % hyp_val) where passThrough are the residual tokens of hyp_name
        % after removing the target object name. In most cases the residual
        % token will be a hyperparameter name, but if you're using a
        % pipeline of pipelines then the residual token could be another
        % parameter of the form class_param, in which case the function
        % recurses.
        function set_params(obj, hyp_name, hyp_val)
            hyp_name = strsplit(hyp_name,'__');
            for i = 1:length(obj.transformers)
                if strcmp(hyp_name{1}, obj.transformer_names{i})
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.transformers{i}.set_params(passThrough, hyp_val);
                    return
                end
            end
            for i = 1:length(obj.estimator)
                if strcmp(hyp_name{1}, obj.estimator_name)
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.estimator.set_params(passThrough, hyp_val);
                    return
                end
            end
        end
        
        function obj = set_transformer(obj,transformers)
            obj.transformer = {};
            obj.transformer_names = {};
            
            for i = 1:length(transformers)
                assert(isa(transformers{i}{2}, 'Transformer'), 'All steps must be transformers');
                
                obj.transformer_names{end+1} = transformers{i}{1};
                obj.transformers{end+1} = copy(transformers{i}{2});
            end
            
            obj.isFitted = false;
        end
        
        function obj = set_estimator(obj,estimator)       
            assert(isa(estimator{2}, 'Estimator'), 'estimator must be type Estimator');

            obj.estimator_name = estmator{1};
            obj.estimator = copy(estimator{2});
            
            obj.isFitted = false;
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
