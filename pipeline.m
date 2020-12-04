classdef pipeline < fmriDataPredictor & fmriDataTransformer
    properties 
        verbose = false;
    end
    properties (SetAccess = private)
        transformers = [];
        predictor = [];
        
        isFitted = false;
        
        fitTime = -1
    end
    
    properties (Access = {?fmriDataTransformer, ?fmriDataPredictor})
        hyper_params = {};
    end
    
    methods
        function obj = pipeline(steps, varargin)
            for i = 1:length(steps) - 1
                assert(isa(steps{i}, 'fmriDataTransformer'), 'All but last step must be transformers');
                
                obj.transformers{end+1} = steps{i};
            end
            if isa(steps{end},'fmriDataPredictor')
                obj.predictor = steps{end};
            elseif isa(steps{end},'fmriDataTransformer')
                obj.transformers{end+1} = steps{end};
            else
                error('Last step must be a transformer or a predictor');
            end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'verbose'
                            obj.verbose = varargin{i}+1;
                    end
                end
            end
            
            % set hyperparameters to match whatever the constituent class'
            % hyperparameters are
            params = {};
            for i = 1:length(obj.transformers)
                these_params = obj.transformers{i}.get_params();
                params = [params, cellfun(@(x1)([class(obj.transformers{i}), '__', x1]), ...
                   these_params, 'UniformOutput', false)];
            end
            if ~isempty(obj.predictor)
                warning('off','bayesOptClf:get_params')
                these_params = obj.predictor.get_params();
                warning('on','bayesOptClf:get_params');
                params = [params, cellfun(@(x1)([class(obj.predictor), '__', x1]), these_params, 'UniformOutput', false)];
            end
            
            obj.hyper_params = params;
            
            % redundant classes among transformers will break
            % implementation of pipeline.set_hyp(), so check for it.
            transformerClasses = cellfun(@class, obj.transformers,'UniformOutput',false);
            assert(length(unique(transformerClasses)) == length(transformerClasses), ...
                'Repeated use of the same transformer class is not supported. Consider wrapping transformer replicates in a pipeline and nesting pipelines as a workaround.');
        end
        
        % fit all transformers and any predictors
        function obj = fit(obj, dat, Y)
            t0 = tic;
            for i = 1:length(obj.transformers)
                % output from one transformer is input to the next
                fprintf('Fitting %s\n', class(obj.transformers{i}));
                [obj.transformers{i}, dat] = obj.transformers{i}.fit_transform(dat);            
            end
            if ~isempty(obj.predictor)
                fprintf('Fitting %s\n', class(obj.predictor));
                
                obj.predictor = obj.predictor.fit(dat, Y);
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        % apply all transforms
        function dat = transform(obj, dat)
            for i = 1:length(obj.transformers)
                fprintf('Applying %s\n', class(obj.transformers{i}));
                % output from one transformer is input to the next
                dat = obj.transformers{i}.transform(dat);
            end
        end
        
        % apply all transforms and predict
        function yfit = predict(obj, dat)
            assert(~isempty(obj.predictor), ...
                'This pipeline does not terminate in a predictor. Try pipeline.transform() instead');
            
            dat = obj.transform(dat);
            if ~isempty(obj.predictor)
                fprintf('Applying %s\n', class(obj.predictor));
                yfit = obj.predictor.predict(dat);
            end
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
                if strcmp(hyp_name{1},class(obj.transformers{i}))
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.transformers{i} = obj.transformers{i}.set_hyp(passThrough, hyp_val);
                    return
                end
            end
            for i = 1:length(obj.predictor)
                if strcmp(hyp_name{1},class(obj.predictor))
                    passThrough = strjoin(hyp_name(2:end),'__');
                    obj.predictor = obj.predictor.set_hyp(passThrough, hyp_val);
                    return
                end
            end
        end
        
        function obj = set_transformer(obj,transformers)
            obj.transformers = transformers;
            obj.isFitted = false;
        end
        
        function obj = set_predictor(obj,predictor)
            obj.predictor = predictor;
            obj.isFitted = false;
        end
    end
end