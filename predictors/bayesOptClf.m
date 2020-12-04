% bayesOptClf Create a bayesOptimized predictor
%
%   predictor = bayesOptPredictor(predictor, [cv], [scorer], bayesOptOpts)
%
%   predictor - a predictor with a get_params() and set_hyp() method
%               get_params(), which must allow these kinds of operations,
%               params = predictor.getParams()
%               predictor = predictor.set_hyp(params{1}, newVal)
%               string valued names returned by getParams() define valid
%               values of the name field of bayesOpt optimizable variables
%               subsequently passed to this class.
%
%   cv      - a function handle that takes an fmri_data object and target
%               as input and returns a cvpartition object. Default is 
%               cv = @(dat)cvpartition2(ones(length(dat.Y),1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id)
%
%   scorer  - a function handle that takes a yFit object as input and returns a
%               scalar value loss estimate. Default is get_mse(). yFit
%               objects have yfit, yfit_null and Y properties
%
%   bayesOptOpts - same arguments you would normally supply to bayesopt if
%               invoked directly (see help bayesOpt for details). Note that
%               any optimizableVariable object must have 'Name' set to a
%               value matching those returned by predictor.get_params().
%
%   bayesOptClf methods:
%       fit     - run bayesopt to identify best hyperparameters
%       predict - get prediction using optimally fit hyperparameters
%       
%
% Example ::
%
%   this_dat % an fmri_data_st object with dat.metadata_table.subject_id
%            % indicating subject block membership
%
%   predictor = plsRegressor();
%
%   dims = optimizableVariable('numcomponents',[1,30], 'Type', 'integer', 'Transform', 'log');
%   bayesOptOpts = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
%    'MaxObjectiveEvaluations', 2, 'UseParallel' 0, 'verbose', 0};
%
%   cvpart = @(dat,Y)cvpartition2(ones(Y,1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id);
%   bo = bayesOptClf(predictor, cvpart, @get_mse, bayesOptOpts)
%
%   bo = bo.fit(this_dat, this_dat.Y);
%   yfit = bo.predict(new_dat)

classdef bayesOptClf < fmriDataPredictor
    properties
        bayesOptOpts = [];
        predictor = [];
        cv = @(dat,Y)cvpartition2(ones(length(dat.Y),1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id)
        scorer = @get_mse;
    end
    
    properties (SetAccess = private)
        group_id = [];
        fitTime = -1;
    end
    
    properties (Access = ?fmriDataPredictor)
        hyper_params = {};
    end
    
    methods
        function obj = bayesOptClf(predictor, cv, scorer, bayesOptOpts)
            obj.predictor = predictor;
            if ~isempty(cv), obj.cv = cv; end
            if ~isempty(scorer), obj.scorer = scorer; end
            
            params = predictor.get_params();
            for i = 1:length(bayesOptOpts{1})
                this_param = bayesOptOpts{1}(i);
                
                assert(ismember(this_param.Name, params),...
                    sprintf('optimizableVariable names must match %s.get_params()\n', class(predictor)));
            end
            
            obj.bayesOptOpts = bayesOptOpts;
        end
        
        function obj = fit(obj, dat, Y)  
            t0 = tic;
            % obj = fit(obj, dat, Y) optimizes the hyperparameters of
            % obj.predictor using data in fmri_data object dat and target vector
            % Y.
            lossFcn = @(hyp)(obj.lossFcn(hyp, dat, Y));
            bayesOptObj = bayesopt(lossFcn, obj.bayesOptOpts{:});
            this_hyp = bayesOptObj.XAtMinEstimatedObjective;
            
            params = obj.predictor.get_params();
            for i = 1:length(params)
                obj.predictor = obj.predictor.set_hyp(params{i}, this_hyp.(params{i}));
            end
            
            obj.predictor = obj.predictor.fit(dat, Y);
            obj.fitTime = toc(t0);
        end
        
        function yfit = predict(obj, dat)
            % yfit = predict(obj, dat) makes a prediction using optimal
            % hypermaraters of obj.predictor on fmri_data object dat.
            yfit = obj.predictor.predict(dat);
        end
        
        function obj = set_hyp(obj, varargin)
            warning('bayesOptClf:get_params','This function has no hyperparameters to set. To set hyperparameters of obj.predictor call obj.fit() instead.');
        end
        
        function params = get_params(varargin)
            warning('bayesOptClf:get_params','This function should not be optimized. It is an optimizer.');
            params = {};
        end
        
        function loss = lossFcn(obj, this_hyp, dat, Y)
            % set hyperparameters
            params = obj.predictor.get_params();
            for i = 1:length(this_hyp.Properties.VariableNames)
                hypname = this_hyp.Properties.VariableNames{i};
                assert(~isempty(ismember(params, hypname)), ...
                    sprintf('%s is not a valid hyperparameter for bayes optimization', hypname));
                
                obj.predictor = obj.predictor.set_hyp(hypname, this_hyp.(hypname));
            end
            
            % what follows is valid if computing the score across folds and
            % averaging gives the same result as computing the score for
            % each fold individually and averaging those scores. It's valid
            % for MSE for instance, but not r-value. We need to update this
            % funtcion with a crossValScore function that we then average.
            this_cv = crossValPredict(obj.predictor, obj.cv, 'repartOnFit', true, 'n_parallel', 1, 'verbose', false);
            
            % get associated loss
            this_cv = this_cv.do(dat, Y);
            loss = obj.scorer(this_cv);
        end 
    end
end