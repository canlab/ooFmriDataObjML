% crossValScore computes cross validated prediction and evaluates a
% performance metric for each fold, saved in crossValScore.scores.
%
% cvEstimator = crossValScore(estimator, cv, scorer, varargin)
%
% estimator - a baseEstimator type object
%
% cv    - a function handle to a method that takes an (X,Y) value as input,
%           e.g. cv(fmri_data, Y), and returns a cvpartition
%           object. Any fields of fmri_data referenced must be preserved
%           across obj.cat and obj.get_wh_image() invocations.
%
% scorer - a function handle to a yFit scorer that takes a yfit object as
%           input and returns a scalar value estimate of performance. This
%           should be fairly naive and not fold aware. crossValScore will 
%           handle repeated application across CV folds.
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
% 
% ToDo:
%   This needs a major update to support non-kfold partitioning schemes

classdef crossValScore < crossValidator % note this is not a yFit object, only crossValPredict is
    properties
        cvpart = [];
        scorer = [];
    end
    
    properties (SetAccess = private)
        scores = [];
        scores_null = [];
        
        evalTimeScorer = -1;
        evalTimeFits = -1;
    end
    
    properties (Hidden = true)
        % function which when applied to X data object extracts metadata
        % needed by scorer.
        % idx correspond to fold indices
        % Hidden because the interface is pretty hacky and ineligant, so
        % its use should be discouraged.
        scorer_metadata_constructor = @(X, idx)([]);
    end
    
    properties (SetAccess = ?crossValidator)        
        yfit = {};      % predicted scores or category labels, 1 x n_partitions cell array
        yfit_raw = {};  % same as yfit for regression, but raw scores for categorical outcomes
        yfit_null = {};
        
        Y = {};         % observed scores or category labels, 1 x n_partitions cell array
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
                        
            % do coss validation. Dif invocations for parallel vs. serial            
            this_foldEstimator = cell(obj.cvpart.NumTestSets,1);
            for i = 1:obj.cvpart.NumTestSets
                this_foldEstimator{i} = copy(obj.estimator);
            end
            if obj.n_parallel <= 1            
                for i = 1:obj.cvpart.NumTestSets
                    if obj.verbose, fprintf('Evaluating fold %d/%d\n', i, obj.cvpart.NumTestSets); end

                    obj.Y{i} = Y(obj.cvpart.test(i),:);
                    
                    train_Y = Y(~obj.cvpart.test(i),:);
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
                        obj.yfit{i} = this_baseEstimator.decisionFcn(tmp_yfit_raw);
                        
                        assert(length(this_baseEstimator.classLabels) == length(obj.classLabels), ...
                            sprintf('Number of class labels in Y (%d) don''t match number in Y partition (%d). Check that cvpartitioner is appropriately straifying outcomes cross folds.', ...
                                length(this_baseEstimator.classLabels),length(obj.classLabels)));
                        
                        nClasses = length(obj.classLabels);
                        if nClasses > 2
                            resortIdx = zeros(1,nClasses);
                            for j = 1:nClasses % the assertion above ensures fold estimators will have nClasses
                                resortIdx(j) = find(this_baseEstimator.classLabels == obj.classLabels(j));
                            end
                            tmp_yfit_raw = tmp_yfit_raw(:,resortIdx);
                        end
                    else
                        obj.yfit{i} = tmp_yfit_raw;
                    end
                    
                    obj.yfit_raw{i} = tmp_yfit_raw;
                end
            else
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool(obj.n_parallel);
                elseif pool.NumWorkers ~= obj.n_parallel
                    delete(gcp('nocreate')); 
                    parpool(obj.n_parallel);
                end
                [yfit, yfit_raw, tmp_Y] = deal(cell(obj.cvpart.NumTestSets,1));
                
                % obj cannot be classified in parfor loop, and we need Y
                % assigned or obj.classLabels breaks within the parfor
                % loop.
                for i = 1:obj.cvpart.NumTestSets
                    obj.Y{i} = Y(obj.cvpart.test(i),:);
                end
                
                parfor i = 1:obj.cvpart.NumTestSets                    
                    train_Y = Y(~obj.cvpart.test(i),:);
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
                
                obj.yfit_raw = yfit_raw;
                obj.yfit = yfit;
            end
                        
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
            obj.yfit_null = {};
            obj.scores_null = zeros(obj.cvpart.NumTestSets, size(Y{1},2));
            for i = 1:obj.cvpart.NumTestSets                
                tmp_yfit_raw = obj.foldEstimator{i}.score_null(sum(obj.cvpart.TestSize(i)));
                
                %{
                % any score_samples columns will need to be resorted to
                % match classLabels here since they may differ across
                % foldEstimators.
                %
                % note: this code has not been updated since crossValScore
                % Y, yfit and yfit_null properties were changed to cell
                % arrays from vectors
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
                fold_Y = obj.Y{i};
                
                yfit = manual_yFit(fold_Y, fold_yfit_null, fold_yfit_raw);
                yfit.classLabels = obj.classLabels;
                yfit.set_null(fold_Y); % superfluous but needed to avoid errors if using normalized metrics.
                yfit.metadata = obj.scorer_metadata_constructor(obj, obj.cvpart.test(i));
                
                obj.scores_null(i,:) = obj.scorer(yfit);
                obj.yfit_null{i} = fold_yfit_null;
            end
        end
        
        
        function eval_score(obj)
            assert(obj.is_done, 'Please run obj.do first');
            
            t0 = tic;
            k = unique(obj.cvpart.NumTestSets);
            
            obj.scores = zeros(k,size(obj.Y{1},2));
            for i = 1:k
                fold_Y = obj.Y{i};
                fold_yfit_raw = obj.yfit_raw{i};
                fold_yfit = obj.yfit{i};
                
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
                    yfit.set_null(obj.yfit_null{i});
                end
                
                yfit.metadata = obj.scorer_metadata_constructor(obj, obj.cvpart.test(i));
                
                obj.scores(i,:) = obj.scorer(yfit);
            end
            obj.evalTimeScorer = toc(t0);
        end
                
        function repartition(obj)
            obj.cvpart = obj.cvpart.repartition;
            
            obj.yfit = [];
            obj.yfit_null = [];
            obj.evalTime = -1;
            obj.is_done = false;

            obj.evalTimeFits = -1;
            obj.evalTimeScorer = -1;
            obj.scores = [];
            obj.scores_null = [];
        end
        
        %% dependent methods
        function set.cvpart(obj, cvpart)
           obj.cvpart = cvpart;
           obj.yfit = [];
           obj.yfit_null = [];
           obj.evalTime = -1;
           obj.is_done = false;
           obj.repartOnFit = false;
        
           obj.evalTimeFits = -1;
           obj.evalTimeScorer = -1;
           obj.scores = [];
           obj.scores_null = [];
        end
        
        function set.scorer(obj, scorer)
           obj.scorer = scorer;
           obj.evalTime = -1;
           obj.evalTimeScorer = -1;
        end
        
        %% convenience functions
        
        function varargout = plot(obj, varargin)
            % conversion to a crossValPredict object is a shortcut, but
            % won't work for crossValidators with overlapping partitions.
            % crossValScore.plot() could be rewritten to handle this
            % situation, and break from the crossValPredict implementation, 
            % if this ends up being needed
            assert(isa(obj.cvpart, 'cvpartition'), ...
                'crossValScore.plot() for objects with overlapping partitions has not been implemented. Try plotting manually.');
            
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
    
