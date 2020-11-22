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
    
    methods
        function obj = pipeline(steps, varargin)
            for i = 1:length(steps) - 1
                assert(isa(steps{i}, 'fmriDataTransformer'), 'All but last step must be transformers');
                
                obj.transformers{end+1} = steps{i};
            end
            if isa(steps{end},'fmriDataTransformer')
                obj.transformers{end+1} = steps{end};
            elseif isa(steps{end},'fmriDataPredictor')
                obj.predictor = steps{end};
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
            
            params = [];
            % not yet implemented
            %for i = 1:length(obj.transformers)
            %    these_params = obj.transformers{i}.get_params();            
            %    params = [params,
            %    cellfun(@(x1)([class(obj.transformers{i}), '__', x1]), ...
            %       these_params, 'UniformOutput', false)];
            %end
            if ~isempty(obj.predictor)
                these_params = obj.predictor.get_params();
                params = [params, cellfun(@(x1)([class(obj.predictor), '__', x1]), these_params, 'UniformOutput', false)];
            end
        end
    end
end