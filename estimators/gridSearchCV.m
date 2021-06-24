% gridSearchCV Create a optimized estimator using a grid search for
% hyperparameters
%
%   estimator = gridSearchCV(estimator, gridPoints, [cv], [scorer])
%
%   estimator - an Estimator with a get_params() and set_params()
%               method, which must allow these kinds of operations,
%               params = estimator.getParams()
%               estimator = estimator.set_params(params{1}, newVal)
%               string valued names returned by getParams() define valid
%               values of the name field of bayesOpt optimizable variables
%               subsequently passed to this class.
%
%   gridPoints
%             - table of grid points to evaluate. Each column 
%               represents a variable and each row is a hyperparameter 
%               combination that should be evaluated. Names must match
%               estimator.get_params()
%
%   cv      - a function handle that takes an fmri_data object and target
%               as input and returns a cvpartition object. Default is 
%               cv = @(dat)cvpartition(ones(length(dat.Y),1),'KFOLD', 5).
%               Look into cvpartition2 if you have blocks of dependent data 
%               (e.g. repeated measurements).
%
%   scorer  - a function handle that takes a yFit object as input and returns a
%               scalar value loss estimate. Default is get_mse(). yFit
%               objects have yfit, yfit_null and Y properties
%
% Optional ::
%
%   verbose - true/false. Default: false
%
%   n_parallel 
%           - number of parallel workers to use. Default: 1
%
%
%   gridSearchCV methods:
%       fit     - run gridSearchCV to identify best hyperparameters
%       predict - get prediction using optimally fit hyperparameters
%       
%
% Example ::
%
%   this_dat % an fmri_data_st object with dat.metadata_table.subject_id
%            % indicating subject block membership
%
%   estimator = plsRegressor();
%
%   dims = optimizableVariable('numcomponents',[1,30], 'Type', 'integer', 'Transform', 'log');
%   gridSearchCV = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
%    'MaxObjectiveEvaluations', 2, 'UseParallel' 0, 'verbose', 0};
%
%   cvpart = @(dat,Y)cvpartition2(ones(Y,1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id);
%   bo = bayesOptCV(estimator, cvpart, @get_mse, bayesOptOpts)
%
%   bo = bo.fit(this_dat, this_dat.Y);
%   yfit = bo.predict(new_dat)

classdef gridSearchCV < baseEstimator
    properties
        estimator = [];
        cv = @(dat,Y)cvpartition(ones(length(dat.Y),1),'KFold', 5)
        scorer = [];
        
        verbose = false;
        n_parallel = 1;
    end
    
    properties (SetAccess = immutable)
        gridPoints = [];
    end
    
    properties (SetAccess = private)
        group_id = [];
    end
    
    properties (Access = ?Estimator)
        hyper_params = {};
    end
    
    methods
        function obj = gridSearchCV(estimator, gridPoints, cv, scorer, varargin)
            obj.estimator = copy(estimator);
            if ~isempty(cv), obj.cv = cv; end

            if isempty(scorer)
                if isa(getBaseEstimator(estimator),'modelClf')
                    obj.scorer = @get_hinge_loss;
                elseif isa(getBaseEstimator(estimator),'modelRegressor')
                    obj.scorer = @get_mse;
                else
                    warning('Could not autodetect model type, and no scorer specified. Defaulting to @get_mse.');
                    obj.scorer = @get_mse;
                end
            else
                obj.scorer = scorer;
            end
            
            params = estimator.get_params();
            for i = 1:length(gridPoints.Properties.VariableNames)
                this_param = gridPoints.Properties.VariableNames{i};
                
                assert(ismember(this_param, params),...
                    sprintf('optimizableVariable names must match %s.get_params()\n', class(estimator)));
            end
            
            obj.gridPoints = gridPoints;
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'verbose'
                            obj.verbose = varargin{i+1};
                        case 'n_parallel'
                            obj.n_parallel = varargin{i+1};
                        otherwise
                            warning(sprintf('Option %s unsupported',varargin{i}));
                    end
                end
            end
        end
        
        function obj = fit(obj, dat, Y, varargin)  
            t0 = tic;
            % obj = fit(obj, dat, Y) optimizes the hyperparameters of
            % obj.estimator using data in fmri_data object dat and target vector
            % Y.
            lossFcn = @(hyp)(obj.lossFcn(hyp, dat, Y));
            
            
            loss = zeros(height(obj.gridPoints),1);
            if obj.n_parallel > 1
                parfor (i = 1:height(obj.gridPoints), obj.n_parallel)
                    warning('off', 'cvpartitionMemoryImpl2:updateParams');
                    
                    loss(i) = lossFcn(obj.gridPoints(i,:));
                    if obj.verbose
                        if i == 1
                            warning('on', 'cvpartitionMemoryImpl2:updateParams');
                            names = [obj.gridPoints.Properties.VariableNames, 'Loss'];
                            for j = 1:length(names)
                                fprintf('%s\t|\t',names{j});
                            end
                            fprintf('\n');
                            warning('off', 'cvpartitionMemoryImpl2:updateParams');
                        end
                        disp([table2cell(obj.gridPoints(i,:)), loss(i)]);
                    end
                end
            else
                for i = 1:height(obj.gridPoints)
                    loss(i) = lossFcn(obj.gridPoints(i,:));
                    if obj.verbose
                        if i == 1
                            names = [obj.gridPoints.Properties.VariableNames, 'Loss'];
                            for j = 1:length(names)
                                fprintf('%s\t|\t',names{j});
                            end
                            fprintf('\n');
                        end
                        disp([table2cell(obj.gridPoints(i,:)), loss(i)]);
                    end
                end
            end
            %%
            this_hyp = obj.gridPoints(find(loss == min(loss), 1, 'first'),:);
            
            params = obj.estimator.get_params();
            for i = 1:length(this_hyp.Properties.VariableNames)
                hypname = this_hyp.Properties.VariableNames{i};       
                assert(~isempty(ismember(params, hypname)), ...
                    sprintf('%s is not a valid hyperparameter for %s', hypname, class(obj.estimator)));
                
                obj.estimator.set_params(hypname, this_hyp.(hypname));
            end
            
            obj.estimator.fit(dat, Y, varargin{:});
            obj.fitTime = toc(t0);
        end
        
        
        function yfit_raw = score_samples(obj, dat, varargin)
            yfit_raw = obj.estimator.score_samples(dat, varargin{:});
        end
        
        function yfit_null = score_null(obj, varargin)
            yfit_null = obj.estimator.score_null(varargin{:});
        end
        
        function yfit = predict(obj, varargin)
            assert(obj.isFitted,'Please run obj.fit() before obj.predict()');
            yfit = obj.estimator.predict(varargin{:});
        end
        
        function yfit_null = predict_null(obj, varargin)
            yfit_null = obj.estimator.predict_null(varargin{:});
        end
        
        function set_params(~, varargin)
            warning('bayesOptCV:get_params','This function has no hyperparameters to set. To set hyperparameters of obj.estimator call obj.fit() instead.');
        end
        
        function params = get_params(varargin)
            warning('bayesOptCV:get_params','This function should not be optimized. It is an optimizer.');
            params = {};
        end
        
        function loss = lossFcn(obj, this_hyp, dat, Y)
            % make a copy. crossValScore ultimately makes a copy too, but
            % we should control copy functionality here to ensure changes
            % to crossValScore don't break this function down the line.
            this_estimator = copy(obj.estimator);
            % set hyperparameters
            params = this_estimator.get_params();
            for i = 1:length(this_hyp.Properties.VariableNames)
                hypname = this_hyp.Properties.VariableNames{i};
                assert(~isempty(ismember(params, hypname)), ...
                    sprintf('%s is not a valid hyperparameter for bayes optimization', hypname));
                
                this_estimator.set_params(hypname, this_hyp.(hypname));
            end
            
            this_cv = crossValScore(this_estimator, obj.cv, obj.scorer, 'repartOnFit', true, 'n_parallel', 1, 'verbose', false);
            this_cv.scorer_metadata_constructor = @(cvObj,id)(cvObj.cvpart.grp_id(id));
            
            % get associated loss
            this_cv.do(dat, Y);
            loss = mean(this_cv.scores); % might want to make this flexible (e.g. let user pick median or mode)
            
            delete(this_cv) % not sure if this is needed. Are handles created in this function destroyed implicity after the function exists?
        end 
    end
end
