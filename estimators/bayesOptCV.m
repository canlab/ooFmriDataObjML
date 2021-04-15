% bayesOptCV Create a bayesOptimized estimator
%
%   estimator = bayesOptCV(estimator, [cv], [scorer], bayesOptOpts)
%
%   estimator - an Estimatorm must allow these kinds of operations,
%               params = estimator.getParams()
%               estimator = estimator.set_params(params{1}, newVal)
%               estimator = estimator.fit(X,Y)
%               yfit = estimator.predict(X)
%               string valued names returned by getParams() define valid
%               values of the name field of bayesOpt optimizable variables
%               subsequently passed to this class.
%
%   cv        - a function handle that takes an (X,Y) pair as input and
%               returns a cvpartition object. Default is 
%               cv = @(X,Y)cvpartition(ones(length(Y),1),'KFOLD', 5).
%               Look into cvpartition2 if you have blocks of dependent data 
%               (e.g. repeated measurements).
%
%   scorer  - a function handle that takes a yFit object as input and returns a
%               scalar value loss estimate. Default is get_mse(). yFit
%               objects have yfit, yfit_null and Y properties
%
%   bayesOptOpts - same arguments you would normally supply to bayesopt if
%               invoked directly (see help bayesOpt for details). Note that
%               any optimizableVariable object must have 'Name' set to a
%               value matching those returned by estimator.get_params().
%
%   bayesOptCV methods:
%       fit     - run bayesopt to identify best hyperparameters
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
%   bayesOptOpts = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
%    'MaxObjectiveEvaluations', 2, 'UseParallel' 0, 'verbose', 0};
%
%   cvpart = @(dat,Y)cvpartition2(ones(Y,1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id);
%   bo = bayesOptCV(estimator, cvpart, @get_mse, bayesOptOpts)
%
%   bo = bo.fit(this_dat, this_dat.Y);
%   yfit = bo.predict(new_dat)

classdef bayesOptCV < baseEstimator
    properties
        bayesOptOpts = [];
        estimator = [];
        cv = @(X,Y)cvpartition(ones(length(Y),1),'KFOLD', 5)
        scorer = [];
    end
    
    properties (SetAccess = private)
        group_id = [];
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {};
    end
    
    methods
        function obj = bayesOptCV(estimator, cv, scorer, bayesOptOpts)
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
            for i = 1:length(bayesOptOpts{1})
                this_param = bayesOptOpts{1}(i);
                
                assert(ismember(this_param.Name, params),...
                    sprintf('optimizableVariable names must match %s.get_params()\n', class(estimator)));
            end
            
            
            obj.bayesOptOpts = bayesOptOpts;
        end
        
        function fit(obj, dat, Y, varargin)              
            t0 = tic;
            % fit(obj, dat, Y) optimizes the hyperparameters of
            % obj.estimator using data in fmri_data object dat and target vector
            % Y.
            lossFcn = @(hyp)(obj.lossFcn(hyp, dat, Y));
            bayesOptObj = bayesopt(lossFcn, obj.bayesOptOpts{:});
            this_hyp = bayesOptObj.XAtMinEstimatedObjective;
            
            params = obj.estimator.get_params();
            for i = 1:length(this_hyp.Properties.VariableNames)
                hypname = this_hyp.Properties.VariableNames{i};       
                assert(~isempty(ismember(params, hypname)), ...
                    sprintf('%s is not a valid hyperparameter for %s', hypname, class(obj.estimator)));
                
                obj.estimator.set_params(hypname, this_hyp.(hypname));
            end
            
            obj.estimator.fit(dat, Y, varargin{:});
            obj.fitTime = toc(t0);
            obj.isFitted = true;
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
            
            % get associated loss
            this_cv.do(dat, Y);
            loss = mean(this_cv.scores); % might want to make this flexible (e.g. let user pick median or mode)
            
            delete(this_cv) % not sure if this is needed. Are handles created in this function destroyed implicity after the function exists?
        end 
    end
end
