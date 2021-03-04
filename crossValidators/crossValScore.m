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
        scorer = [];
        scores = [];
        scores_null = [];
        
        evalTimeScorer = -1;
        evalTimeFits = -1;
    end
    
    properties
        cvpart = [];
    end
    
    methods
        function obj = crossValScore(estimator, cv, scorer, varargin)
            obj@crossValidator(estimator, cv, varargin{:});
            
            obj.scorer = scorer;
        end
        
        function obj = do(obj, dat, Y)
            t0 = tic;
            if obj.repartOnFit || isempty(obj.cvpart)
                try
                    obj.cvpart = obj.cv(dat, Y);
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
            obj.yfit_raw = [];
            
            this_foldEstimator = cell(obj.cvpart.NumTestSets,1);
            for i = 1:obj.cvpart.NumTestSets
                this_foldEstimator{i} = copy(obj.estimator);
            end
            if obj.n_parallel <= 1            
                for i = 1:obj.cvpart.NumTestSets
                    if obj.verbose, fprintf('Fold %d/%d\n', i, obj.cvpart.NumTestSets); end

                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(dat,'image_vector')
                        train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                        test_dat = dat.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = dat(~obj.cvpart.test(i),:);
                        test_dat = dat(obj.cvpart.test(i),:);
                    end

                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    obj.yfit_raw = [obj.yfit_raw; this_foldEstimator{i}.score_samples(test_dat, 'fast', true)];
                end
            else
                if ~isempty(gcp('nocreate')), delete(gcp('nocreate')); end
                parpool(obj.n_parallel);
                yfit_raw = cell(obj.cvpart.NumTestSets,1);
                parfor i = 1:obj.cvpart.NumTestSets
                    
                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(dat,'image_vector')
                        train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                        test_dat = dat.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = dat(~obj.cvpart.test(i),:);
                        test_dat = dat(obj.cvpart.test(i),:);
                    end

                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    % we can always make certain assumptions about the train and test space
                    % matching which allows us to use the fast option
                    yfit_raw{i} = this_foldEstimator{i}.score_samples(test_dat, 'fast', true)';
                    
                    if obj.verbose, fprintf('Completed fold %d/%d\n', i, obj.cvpart.NumTestSets); end
                end
                
                obj.yfit_raw = cell2mat(yfit_raw{i});
            end
            
            [~,I] = sort(obj.fold_lbls);
            [~,II] = sort(I);
            obj.yfit_raw = obj.yfit_raw(II,:); 
            
            obj.foldEstimator = this_foldEstimator;
            obj.Y = Y;
            
            obj.evalTime = -1;
            obj.evalTimeFits = toc(t0);
            obj.is_done = true;
            
            obj = obj.eval_score();
            
            % evaluate scores for each fold separately
            obj.evalTime = obj.evalTimeScorer + obj.evalTimeFits;
        end
        
        function obj = do_null(obj, varargin)
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
            
            yfit_raw = zeros(length(Y),1);
            obj.yfit_null = zeros(length(Y),1);
            
            for i = 1:obj.cvpart.NumTestSets                
                yfit_raw(obj.cvpart.test(i)) = ...
                    obj.foldEstimator{i}.score_null(obj.cvpart.TestSize(i));
                
                obj.yfit_null(obj.cvpart.test(i)) = ...
                    obj.foldEstimator{i}.predict_null(obj.cvpart.TestSize(i));
            end
            
            k = unique(obj.cvpart.NumTestSets);
            obj.scores_null = zeros(k,1);
            for i = 1:k
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                fold_yfit_raw = yfit_raw(obj.cvpart.test(i));
                fold_yfit = obj.yfit_null(obj.cvpart.test(i));
                
                yfit = manual_yFit(fold_Y, fold_yfit, fold_yfit_raw);
                
                obj.scores_null(i) = obj.scorer(yfit);
            end
        end
        
        
        function obj = eval_score(obj)
            assert(obj.is_done, 'Please run obj.do first');
            
            t0 = tic;
            k = unique(obj.cvpart.NumTestSets);
            
            obj.scores = zeros(k,1);
            for i = 1:k
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                fold_yfit_raw = obj.yfit_raw(obj.cvpart.test(i),:);
                
                this_estimator = getBaseEstimator(obj.foldEstimator{i});
                if isa(this_estimator, 'modelRegressor')
                    fold_yfit = fold_yfit_raw;
                elseif isa(this_estimator, 'modelClf')
                    fold_yfit = this_estimator.decisionFcn(fold_yfit_raw);
                else
                    error('Unsupported base estimator type');
                end
                yfit = manual_yFit(fold_Y, fold_yfit, fold_yfit_raw);
                
                obj.scores(i) = obj.scorer(yfit);
            end
            obj.evalTimeScorer = toc(t0);
        end
        
        %% set methods
        % these could be modified to be Dependent properties, and likely 
        % should be 
        function set.cvpart(obj, cvpart)
           obj.cvpart = cvpart;
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
        
        function obj = set_scorer(obj, scorer)
           obj.scorer = scorer;
           obj.evalTime = -1;
           obj.evalTimeScorer = -1;
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
    
