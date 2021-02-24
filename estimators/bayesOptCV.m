% bayesOptCV Create a bayesOptimized estimator
%
%   estimator = bayesOptCV(estimator, [cv], [scorer], bayesOptOpts)
%
%   estimator - an fmriDataEstimator with a get_params() and set_hyp()
%               method, which must allow these kinds of operations,
%               params = estimator.getParams()
%               estimator = estimator.set_hyp(params{1}, newVal)
%               string valued names returned by getParams() define valid
%               values of the name field of bayesOpt optimizable variables
%               subsequently passed to this class.
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

classdef bayesOptCV < fmriDataEstimator
    properties
        bayesOptOpts = [];
        estimator = [];
        cv = @(dat,Y)cvpartition(ones(length(dat.Y),1),'KFOLD', 5)
        scorer = @get_mse;
    end
    
    properties (SetAccess = private)
        group_id = [];
        fitTime = -1;
    end
    
    properties (Access = ?fmriDataEstimator)
        hyper_params = {};
    end
    
    methods
        function obj = bayesOptCV(estimator, cv, scorer, bayesOptOpts)
            obj.estimator = estimator;
            if ~isempty(cv), obj.cv = cv; end
            if ~isempty(scorer), obj.scorer = scorer; end
            
            params = estimator.get_params();
            for i = 1:length(bayesOptOpts{1})
                this_param = bayesOptOpts{1}(i);
                
                assert(ismember(this_param.Name, params),...
                    sprintf('optimizableVariable names must match %s.get_params()\n', class(estimator)));
            end
            
            obj.bayesOptOpts = bayesOptOpts;
        end
        
        function obj = fit(obj, dat, Y)  
            t0 = tic;
            % obj = fit(obj, dat, Y) optimizes the hyperparameters of
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
                
                obj.estimator = obj.estimator.set_hyp(hypname, this_hyp.(hypname));
            end
            
            obj.estimator = obj.estimator.fit(dat, Y);
            obj.fitTime = toc(t0);
        end
        
        function yfit = predict(obj, dat, varargin)
            % yfit = predict(obj, dat) makes a prediction using optimal
            % hypermaraters of obj.estimator on fmri_data object dat.
            yfit = obj.estimator.predict(dat, varargin{:});
        end
        
        function obj = set_hyp(obj, varargin)
            warning('bayesOptCV:get_params','This function has no hyperparameters to set. To set hyperparameters of obj.estimator call obj.fit() instead.');
        end
        
        function params = get_params(varargin)
            warning('bayesOptCV:get_params','This function should not be optimized. It is an optimizer.');
            params = {};
        end
        
        function loss = lossFcn(obj, this_hyp, dat, Y)
            % set hyperparameters
            params = obj.estimator.get_params();
            for i = 1:length(this_hyp.Properties.VariableNames)
                hypname = this_hyp.Properties.VariableNames{i};
                assert(~isempty(ismember(params, hypname)), ...
                    sprintf('%s is not a valid hyperparameter for bayes optimization', hypname));
                
                obj.estimator = obj.estimator.set_hyp(hypname, this_hyp.(hypname));
            end
            
            this_cv = crossValScore(obj.estimator, obj.cv, obj.scorer, 'repartOnFit', true, 'n_parallel', 1, 'verbose', false);
            
            % get associated loss
            this_cv = this_cv.do(dat, Y);
            loss = mean(this_cv.scores); % might want to make this flexible (e.g. let user pick median or mode)
        end 
    end
    
    methods (Access = {?crossValidator, ?fmriDataTransformer, ?fmriDataEstimator})
        function obj = compress(obj)
            obj.estimator = obj.estimator.compress();
        end
    end
end
