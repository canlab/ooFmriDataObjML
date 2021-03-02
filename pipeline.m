classdef pipeline < Estimator & Transformer
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
    
    properties (Access = {?Transformer, ?Estimator})
        hyper_params = {};
    end
    
    methods
        function obj = pipeline(steps, varargin)
            for i = 1:length(steps) - 1
                assert(isa(steps{i}{2}, 'Transformer'), 'All but last step must be transformers');
                
                obj.transformer_names{end+1} = steps{i}{1};
                obj.transformers{end+1} = steps{i}{2};
            end
            if isa(steps{end}{2},'Estimator')
                obj.estimator_name = steps{end}{1};
                obj.estimator = steps{end}{2};
            elseif isa(steps{end}{2},'Transformer')
                obj.transformer_names{end+1} = steps{end}{1};
                obj.transformers{end+1} = steps{end}{2};
            else
                error('Last step must be a transformer or a estimator');
            end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'verbose'
                            obj.verbose = varargin{i}+1;
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
        function obj = fit(obj, dat, Y)
            t0 = tic;
            for i = 1:length(obj.transformers)
                % output from one transformer is input to the next
                fprintf('Fitting %s\n', obj.transformer_names{i});
                [obj.transformers{i}, dat] = obj.transformers{i}.fit_transform(dat);            
            end
            if ~isempty(obj.estimator)
                fprintf('Fitting %s\n', obj.estimator_name);
                
                obj.estimator = obj.estimator.fit(dat, Y);
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        % apply all transforms
        function dat = transform(obj, dat, varargin)
            for i = 1:length(obj.transformers)
                fprintf('Applying %s\n', obj.transformer_names{i});
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
                fprintf('Applying %s\n', obj.estimator_name);
                yfit_raw = obj.estimator.score_samples(dat);
            end
        end
        
        % apply all transforms and predict
        function yfit = predict(obj, dat, varargin)
            assert(~isempty(obj.estimator), ...
                'This pipeline does not terminate in a estimator. Try pipeline.transform() instead');
            
            dat = obj.transform(dat, varargin{:});
            if ~isempty(obj.estimator)
                fprintf('Applying %s\n', obj.estimator_name);
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
        
        % finds object to modify and calls its obj.set_hyp(passThrough,
        % hyp_val) where passThrough are the residual tokens of hyp_name
        % after removing the target object name. In most cases the residual
        % token will be a hyperparameter name, but if you're using a
        % pipeline of pipelines then the residual token could be another
        % parameter of the form class_param, in which case the function
        % recurses.
        function obj = set_hyp(obj, hyp_name, hyp_val)
            hyp_name = strsplit(hyp_name,'__');
            for i = 1:length(obj.transformers)
                if strcmp(hyp_name{1}, obj.transformer_names{i})
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.transformers{i} = obj.transformers{i}.set_hyp(passThrough, hyp_val);
                    return
                end
            end
            for i = 1:length(obj.estimator)
                if strcmp(hyp_name{1}, obj.estimator_name)
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.estimator = obj.estimator.set_hyp(passThrough, hyp_val);
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
                obj.transformers{end+1} = transformers{i}{2};
            end
            
            obj.isFitted = false;
        end
        
        function obj = set_estimator(obj,estimator)       
            assert(isa(estimator{2}, 'Estimator'), 'estimator must be type Estimator');

            obj.estimator_name = estmator{1};
            obj.estimator = estimator{2};
            
            obj.isFitted = false;
        end
    end
end
