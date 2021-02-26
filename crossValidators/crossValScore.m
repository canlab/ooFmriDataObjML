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
%   yfit    - most recent fitted values (cross validated)
%   Y       - most recent observed values
%   yfit_null - null predictions (cross validated)
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

classdef crossValScore < crossValPredict    
    properties (SetAccess = private)
        scorer = [];
        scores = [];
        scores_null = [];
        
        evalTimeScorer = -1;
        evalTimeFits = -1;
    end
    
    methods
        function obj = crossValScore(estimator, cv, scorer, varargin)
            obj@crossValPredict(estimator, cv, varargin{:});
            
            obj.scorer = scorer;
        end
        
        function obj = do(obj, dat, Y)
            t0 = tic;
            obj = do@crossValPredict(obj, dat, Y);
            obj.evalTime = -1;
            obj.evalTimeFits = toc(t0);
            
            obj = obj.eval_score();
            
            % evaluate scores for each fold separately
            obj.evalTime = obj.evalTimeScorer + obj.evalTimeFits;
        end
        
        function obj = do_null(obj, varargin)
            
            obj = do_null@crossValPredict(obj, varargin{:});
            
            for i = 1:obj.cvpart.NumTestSets     
                fold_yfit = obj.yfit_null(obj.cvpart.test(i));
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                yfit = manual_yFit(fold_Y, fold_yfit);
                
                obj.scores_null(i) = obj.scorer(yfit);
            end
        end
        
        function obj = set_cvpart(obj, cvpart)
           obj = set_cvpart@crossValPredict(obj,cvpart);
           
           obj.evalTimeFits = -1;
           obj.evalTimeScorer = -1;
           obj.scores = [];
           obj.scores_null = [];
        end
        
        function obj = eval_score(obj)
            assert(obj.is_done, 'Please run obj.do first');
            
            t0 = tic;
            k = unique(obj.cvpart.NumTestSets);
            obj.scores = zeros(k,1);
            for i = 1:k
                fold_yfit = obj.yfit(obj.cvpart.test(i));
                fold_Y = obj.Y(obj.cvpart.test(i));
                
                yfit = manual_yFit(fold_Y, fold_yfit);
                
                obj.scores(i) = obj.scorer(yfit);
            end
            obj.evalTimeScorer = toc(t0);
        end
        
        function obj = set_scorer(obj, scorer)
           obj.scorer = scorer;
           obj.evalTime = -1;
           obj.evalTimeScorer = -1;
        end
        
        function varargout = plot(obj, varargin)
            varargout = plot@crossValPredict(obj, varargin{:});
            this_title = sprintf('Mean score = %0.3f', mean(obj.scores));
            if ~isempty(obj.scores_null)
                this_title = {this_title, sprintf('Mean null score = %0.3f', mean(obj.scores_null))};
            end
            title(this_title);
        end
    end
end
    
