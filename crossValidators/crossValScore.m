% crossValScore computes cross validated prediction and evaluates a
% performance metric for each fold, saved in crossValScore.scores.
% inherits crossValPredict, so see help for crossValPredict as well in 
% case this help is out of date.
%
% cvEstimator = crossValScore(estimator, cv, varargin)
%
% estimator - an fmriDataEstimator object
%
% cv    - a function handle to a method that takes an fmri_data and target
%           value as input, cv(fmri_data, Y) and returns a cvpartition
%           object. Any fields of fmri_data referenced must be preserved
%           across obj.cat and obj.get_wh_image() invocations.
%
% scorer - a function handle to a yFit scorer that takes a yfit object as
%           input and returns a scalar value estimate of performance. This
%           should not be fold aware. crossValScore will handle repeated
%           application across CV folds.
%
% (optional)
%
% repartOnFit - whether cv should be reinitialized when fitting. To
%         use a predetermined cv object for fitting simply provide a
%         cvEstimator = cvEstimator.set_cvpart(cvpartition);
%         after initialization. Useful for classifier comparison.
%
% verbose  - whether to display fold information in cross validation
%
% n_paralllel - number of threads to use for parallelization
%
% crossValScore properties:
%   cvpart  - cvpartition object used during last fit call
%   estimator
%           - classifier used in last fit call
%   scorer  - scorer used to compute scores
%   scores  - most recent score
%   yfit_raw
%           - most recent fitted values (cross validated)
%   yfit    - most recently predicted values (cross validated, convert to 
%               crossValPredict to populate)
%   Y       - most recent observed values
%   yfit_null 
%           - null predictions (cross validated)
%   foldEstimator
%           - estimator objects use for each fold in fit() call. Useful
%               when hyperparameters differ across folds
%
% crossValScore methods:
%   do      - crossValScore = cvEstimator.fit(fmri_data, Y) performs
%               cross validated predictions using fmri_data to predict Y.
%               Unlike crossValPredict, crossValScore then computes scores
%               for each fold using obj.scorer which are saved in 
%               obj.scores. If the scorer changes between obj.do()
%               invocations, the Y predictions of previous calls are reused
%               for computational efficiency. If the partitioning scheme
%               changes (e.g. by invoking obj.set_cvpart or
%               obj.repartition) then new yfit predictions are computed.
%   do_null - fits null model with cross validation
%   set_cvpart 
%           - sets cvpart manually (useful for reusing cv folds of other
%               training run from a different estimator/classifier)
%   repartition
%           - resets the cvpartition object.
%   set_scorer 
%           - sets scorer manually if you're trying to change this. Resets
%               properties set by old (now obsolete) scorer like scores.
%               You can evaluate old fits using new scorers using this
%               method but if you wish to evaluate new fits you should also
%               invoke obj.repartition() or obj.set_cvpart(cvpartition)
%               with an appropriate cvpartition object.
%

classdef crossValScore < crossValidator & yFit
    properties (SetAccess = private)
        scores = [];
        scores_null = [];
        
        evalTimeScorer = -1;
        evalTimeFits = -1;
    end
    
    properties (Dependent)
        cvpart;
        scorer;
    end
    
    properties (Hidden = true)
        % function which when applied to X data object extracts metadata
        % needed by scorer.
        % idx correspond to fold indices
        % Hidden because the interface is pretty hacky in ineligant.
        scorer_metadata_constructor = @(X, idx)([]);
    end
    
    properties (Hidden = true, Access = private)
        cvpart0 = [];
        scorer0 = [];
    end
    
    methods
        function obj = crossValScore(estimator, cv, scorer, varargin)
            obj@crossValidator(estimator, cv, varargin{:});
            
            obj.scorer = scorer;
        end
        
        function obj = do(obj, X, Y)
            t0 = tic;
            if obj.repartOnFit || isempty(obj.cvpart)
                try
                    obj.cvpart = obj.cv(X, Y);
                catch e
                    e = struct('message', sprintf([e.message, ...
                        '\nError occurred invoking %s.cv. Often this is due to misspecified ',...
                        'input or cvpartitioners.\nMake sure you''re using features ',...
                        'objects if you need them for passing dependence structures.'], class(obj)), 'identifier', e.identifier, 'stack', e.stack);
                    rethrow(e)
                end
            end
            
            % reformat cvpartition and save as a vector of labels. We will
            % use this later to sort the results we get back from each CV
            % fold
            obj.fold_lbls = zeros(length(Y),1);
            for i = 1:obj.cvpart.NumTestSets
                obj.fold_lbls(obj.cvpart.test(i)) = i;
            end
            
            % do coss validation. Dif invocations for parallel vs. serial
            obj.Y = Y;
            [obj.yfit, obj.yfit_raw] = deal([]);
            
            this_foldEstimator = cell(obj.cvpart.NumTestSets,1);
            for i = 1:obj.cvpart.NumTestSets
                this_foldEstimator{i} = copy(obj.estimator);
            end
            if obj.n_parallel <= 1            
                for i = 1:obj.cvpart.NumTestSets
                    if obj.verbose, fprintf('Fold %d/%d\n', i, obj.cvpart.NumTestSets); end

                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(X,'image_vector')
                        train_dat = X.get_wh_image(~obj.cvpart.test(i));
                        test_dat = X.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = X(~obj.cvpart.test(i),:);
                        test_dat = X(obj.cvpart.test(i),:);
                    end

                    
                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    tmp_yfit_raw = this_foldEstimator{i}.score_samples(test_dat, 'fast', true);

                    % any score_samples columns will need to be resorted to
                    % match classLabels here since they may differ across
                    % foldEstimators.
                    this_baseEstimator = getBaseEstimator(this_foldEstimator{i});
                    if isa(this_baseEstimator, 'modelClf')
                        obj.yfit = [obj.yfit; this_baseEstimator.decisionFcn(tmp_yfit_raw)];
                        
                        assert(length(this_baseEstimator.classLabels) == length(obj.classLabels), ...
                            'Number of class labels in Y don''t match number in Y partition. Check that cvpartitioner is appropriately straifying outcomes cross folds.');
                        
                        nClasses = length(obj.classLabels);
                        if nClasses > 2
                            resortIdx = zeros(1,nClasses);
                            for j = 1:nClasses % the assertion above ensures fold estimators will have nClasses
                                resortIdx(j) = find(this_baseEstimator.classLabels == obj.classLabels(j));
                            end
                            tmp_yfit_raw = tmp_yfit_raw(:,resortIdx);
                        end
                    else
                        obj.yfit = [obj.yfit; tmp_yfit_raw];
                    end
                    
                    obj.yfit_raw = [obj.yfit_raw; tmp_yfit_raw];
                end
            else
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool(obj.n_parallel);
                elseif pool.NumWorkers ~= obj.n_parallel
                    delete(gcp('nocreate')); 
                    parpool(obj.n_parallel);
                end
                yfit_raw = cell(obj.cvpart.NumTestSets,1);
                parfor i = 1:obj.cvpart.NumTestSets
                    
                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(X,'image_vector')
                        train_dat = X.get_wh_image(~obj.cvpart.test(i));
                        test_dat = X.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = X(~obj.cvpart.test(i),:);
                        test_dat = X(obj.cvpart.test(i),:);
                    end

                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    % we can always make certain assumptions about the train and test space
                    % matching which allows us to use the fast option
                    tmp_yfit_raw = this_foldEstimator{i}.score_samples(test_dat, 'fast', true);
                    
                    % any score_samples columns will need to be resorted to
                    % match classLabels here since they may differ across
                    % foldEstimators.
                    this_baseEstimator = getBaseEstimator(this_foldEstimator{i});
                    if isa(this_baseEstimator, 'modelClf')
                        assert(length(this_baseEstimator.classLabels) == length(obj.classLabels), ...
                            'Number of class labels in Y don''t match number in Y partition. Check that cvpartitioner is appropriately straifying outcomes cross folds.');
                        yfit{i} = this_baseEstimator.decisionFcn(tmp_yfit_raw);

                        nClasses = length(obj.classLabels);
                        if nClasses > 2
                            resortIdx = zeros(1,nClasses);
                            for j = 1:nClasses % the assertion above ensures fold estimators will have nClasses
                                resortIdx(j) = find(this_baseEstimator.classLabels == obj.classLabels(j));
                            end
                            tmp_yfit_raw = tmp_yfit_raw(:,resortIdx);
                        end
                    else
                        yfit{i} = tmp_yfit_raw;
                    end
                    
                    yfit_raw{i} = tmp_yfit_raw;
                    this_foldEstimator{i} = this_foldEstimator{i}; % propogates modified handle object outside parfor loop
                    
                    if obj.verbose, fprintf('Completed fold %d/%d\n', i, obj.cvpart.NumTestSets); end
                end
                
                obj.yfit_raw = cell2mat(yfit_raw);
                obj.yfit = vertcat(yfit{:});
            end
            
            % resort from fold order to original order
            [~,I] = sort(obj.fold_lbls);
            [~,II] = sort(I);
            obj.yfit_raw = obj.yfit_raw(II,:); 
            obj.yfit = obj.yfit(II,:);
            
            obj.foldEstimator = this_foldEstimator;
            
            obj.evalTime = -1;
            obj.evalTimeFits = toc(t0);
            obj.is_done = true;
            
            obj.eval_score();
            
            % evaluate scores for each fold separately
            obj.evalTime = obj.evalTimeScorer + obj.evalTimeFits;
        end
        
        function obj = do_null(obj, varargin)
            assert(obj.is_done, 'Please run obj.do() first.');
            
            if isempty(obj.cvpart)
                obj.cvpart = obj.cv(varargin{:});
                warning('cvpart not found. null predictions are not valid for yfit obtained with subsequent do() invocations');
            end
            
            if isempty(varargin)
                Y = obj.Y;
            else
                if ~isempty(obj.Y)
                    warning('obj.Y not empty, please use do_null() instead of do_null(~,Y) for best results');
                end
                
                obj.Y = varargin{2};
                Y = obj.Y;
            end
                        
            % compute null scores
            obj.yfit_null = [];
            obj.scores_null = zeros(1, obj.cvpart.NumTestSets);
            for i = 1:obj.cvpart.NumTestSets                
                tmp_yfit_raw = obj.foldEstimator{i}.score_null(sum(obj.cvpart.TestSize(i)));
                
                %{
                % any score_samples columns will need to be resorted to
                % match classLabels here since they may differ across
                % foldEstimators.
                this_baseEstimator = getBaseEstimator(obj.foldEstimator{i});
                if isa(this_baseEstimator, 'modelClf')
                        assert(length(this_baseEstimator.classLabels) == length(obj.classLabels), ...
                            'Number of class labels in Y don''t match number in Y partition. Check that cvpartitioner is appropriately straifying outcomes cross folds.');
                        
                        nClasses = length(obj.classLabels);
                        if nClasses > 2
                            resortIdx = zeros(1,nClasses);
                            for j = 1:nClasses % the assertion above ensures fold estimators will have nClasses
                                resortIdx(j) = find(this_baseEstimator.classLabels == obj.classLabels(j));
                            end
                            tmp_yfit_raw = tmp_yfit_raw(:, resortIdx);
                        end
                end
                %}
                
                fold_yfit_raw = tmp_yfit_raw;
                fold_yfit_null = obj.foldEstimator{i}.predict_null(sum(obj.cvpart.TestSize(i)));
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                yfit = manual_yFit(fold_Y, fold_yfit_null, fold_yfit_raw);
                yfit.classLabels = obj.classLabels;
                yfit.set_null(fold_Y); % superfluous but needed to avoid errors if using normalized metrics.
                yfit.metadata = obj.scorer_metadata_constructor(obj, obj.cvpart.test(i));
                
                obj.scores_null(i) = obj.scorer(yfit);
                obj.yfit_null = [obj.yfit_null; fold_yfit_null];
            end
            [~,I] = sort(obj.fold_lbls);
            [~,II] = sort(I);
            obj.yfit_null = obj.yfit_null(II);
        end
        
        
        function eval_score(obj)
            assert(obj.is_done, 'Please run obj.do first');
            
            t0 = tic;
            k = unique(obj.cvpart.NumTestSets);
            
            obj.scores = zeros(k,1);
            for i = 1:k
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                fold_yfit_raw = obj.yfit_raw(obj.cvpart.test(i),:);
                fold_yfit = obj.yfit(obj.cvpart.test(i),:);
                
                this_estimator = getBaseEstimator(obj.foldEstimator{i});
                if isa(this_estimator, 'modelRegressor')
                    yfit = manual_yFit(fold_Y, fold_yfit, fold_yfit_raw);
                elseif isa(this_estimator, 'modelClf')
                    yfit = manual_yFit(fold_Y, fold_yfit, fold_yfit_raw);
                    yfit.classLabels = obj.classLabels;
                else
                    error('Unsupported base estimator type');
                end
                
                if ~isempty(obj.yfit_null)
                    yfit.set_null(obj.yfit_null(obj.cvpart.test(i)));
                end
                
                yfit.metadata = obj.scorer_metadata_constructor(obj, obj.cvpart.test(i));
                
                obj.scores(i) = obj.scorer(yfit);
            end
            obj.evalTimeScorer = toc(t0);
        end
                
        function obj = repartition(obj)
            obj.cvpart = obj.cvpart.repartition;
        end
        
        %% dependent methods
        function set.cvpart(obj, cvpart)
           obj.cvpart0 = cvpart;
           obj.yfit = [];
           obj.yfit_null = [];
           obj.evalTime = -1;
           obj.fold_lbls = [];
           obj.is_done = false;
        
           obj.evalTimeFits = -1;
           obj.evalTimeScorer = -1;
           obj.scores = [];
           obj.scores_null = [];
        end
        
        function val = get.cvpart(obj)
            val = obj.cvpart0;
        end
        
        function set.scorer(obj, scorer)
           obj.scorer0 = scorer;
           obj.evalTime = -1;
           obj.evalTimeScorer = -1;
        end
        
        function val = get.scorer(obj)
            val = obj.scorer0;
        end
        
        %% convenience functions
        function varargout = plot(obj, varargin)
            warning('off','crossValidator:crossValPredict');
            cvPred = crossValPredict(obj);
            warning('on','crossValidator:crossValPredict');
            varargout{:} = cvPred.plot(varargin{:});
            
            this_title = sprintf('Mean score = %0.3f', mean(obj.scores));
            if ~isempty(obj.scores_null)
                this_title = {this_title, sprintf('Mean null score = %0.3f', mean(obj.scores_null))};
            end
            title(this_title);
        end
    end
end
    
