% linearSvmClf a linearModelEstimator that fits a linear support
%   vector machine classification model to data (X,Y) using the C (slack)
%   parameterization by default and matlab's fitclinear SVM implementation.
%
% estimator = linearSvmRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include intercept in univariate model. 
%                   Default: true;
%
%   'C'         - Slack parameter. To use Lambda specification leave this 
%                   unset and specify Lambda parameter in fitclinearOpts. 
%                   Default C is used when neither is provided. Default: 1
%
%   'epsilon'   - Half width of epsilon intensive band. Can also be
%                   specified as fitclinearOpts argument. See help
%                   fitclinearOpts for default value.
%
%   'regularization'
%               - 'ridge', 'lasso', or 'none'
%
%   'fitclinearOpts'
%               - cell array of options to pass through to firlinear. See
%                   help fitrlinear for details. 
%                 Note: fitclinear uses the Lambda parameterization. If 
%                   you specify a Lambda parameter here instead of a C 
%                   parameter in the linearSvmClf constructor then 
%                   that Lambda parameterization will be used. If you 
%                   specify both Lambda in fitclinearOpts and C in the 
%                   linearSvmClf constructor then a warning will be 
%                   thrown and C will supercede Lambda.
%                 Note: fitclinearOpts has the option of performing
%                   hyperparameter optimization internally via kfold, hold
%                   out or custom cvpartition specification. This is not
%                   supported via linearSvmClf. Wrap linearSvmClf in a
%                   hyperparameter optimization object like bayesOptCV
%                   instead.
%                   
%
%   Note: both lambda and C are hyperparameters, but you should NOT try to
%   optimize both at the same time. Pick one parameterization only.
%
%   Note: fitclinear has a bunch optimization routines built in. These 
%   have not yet been tested. It may be useful for lasso parameter
%   optimization in particular to take advantage of the LARS algorithm,
%   but otherwise its use is discouraged and you are encouraged to instead
%   wrap this routine in a bayesOptCV or gridSearchCV optimizer.
classdef linearSvmClf < linearModelEstimator & modelClf
    properties
        fitclinearOpts = {};
        C = [];
    end
    
    properties (Dependent)
        learner;
        intercept;
        scoreFcn;
    end
    
    properties (Access = private, Hidden)
        % we need to store scoreFcn in both the fitclinear format (which is
        % either a function handle or a string) and in a pure function
        % handle format. To avoid confusion we keep the pure function
        % handle format hidden from the user. This should never be accessed
        % directly though. Always call by invoking scoreFcn, which will use
        % it's get method to pull this data.
        scoreFcn0 = @(x1)(x1);
    end
    
    properties (Dependent, SetAccess = ?modelEstimator)
        lambda;
        regularization;
    end  
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
        
        % CV_funhan = [];
        
        % this is a hacky way of implementing a ternary operator.
        % 1     if x1 true
        % -1    if x1 false
        % I've made attempts to keep methods robust with respect to [0,1]
        % or [-1,1] coding, but just in case we're going to try to use
        % [-1,1] which is standard for SVMs.
        decisionFcn = @(x1)subsref([-1; 1], substruct('()', {(x1 > 0) + 1}));
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'intercept', 'C', 'lambda', 'regularization'};
    end
          
    
    methods
        function obj = linearSvmClf(varargin)
            % defaults
            scoreFcn = 'none';
            intercept = true;
            learner = 'svm';
            regularization = 'none';
            
            fitclinearOpts_idx = find(strcmp(varargin,'fitclinearOpts'));
            if ~isempty(fitclinearOpts_idx)
                obj.fitclinearOpts = varargin{fitclinearOpts_idx + 1};
            end
            
            for i = 1:length(obj.fitclinearOpts)
                if ischar(obj.fitclinearOpts{i})
                    switch(obj.fitclinearOpts{i})
                        case 'ScoreTransform'
                            scoreFcn = obj.fitclinearOpts{i + 1};
                        case 'FitBias'
                            intercept = logical(varargin{i+1});
                        case 'Learner'
                            learner = varargin{i+1};
                        case 'Regularization'
                            regularization = varargin{i+1};
                        case 'KFold'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                        case 'CVPartition'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                        case 'Holdout'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                    end
                end
            end
            % we don't let fitclinearOpts set these directly because
            % setting these in turn will modify fitclineearOpts, and who
            % knows what kind of strange behavior that feedback may cause
            % down the line. Better to have it in two separate invocations.
            obj.scoreFcn = scoreFcn;
            obj.intercept = intercept;
            obj.learner = learner;
            obj.regularization = regularization;
            
            % by parsing fitclinearOpts first we give override priority to
            % arguments passed directly to linearSvmClf, which we parse now
            % and will overwrite anything we did earlier if it differs.
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'intercept'
                            obj.intercept = varargin{i+1};
                        case 'C'
                            obj.C = varargin{i+1};                         
                            if any(strcmp(obj.fitclinearOpts, 'Lambda'))
                                warning('Will override fitclinearOpts Lambda = %0.3f, using C = %0.3f',obj.lambda, obj.C);
                            end
                        case 'regularization'
                            obj.regularization = varargin{i+1};
                        case 'fitclinearOpts'
                            continue;
                        otherwise
                            warning('Option %s not supported', varargin{i});
                    end
                end
            end
            
            if ~any(strcmp(obj.fitclinearOpts, 'Lambda'))
                obj.C = 1;
            end
            
            % see note above check_cv_params definition below
            % obj = obj.check_cv_params();
        end
        
        function obj = fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            
            if ~isempty(obj.C) % if C parameter is in use, set lambda accordingly
                % setting lambda erases C (since outside of this specific
                % case, you can't reconcile the two, since you don't know
                % n, and instead C is erased.
                C = obj.C;
                obj.lambda = C/length(Y);
                obj.C = C;
            end
            
            % see comment above check_cv_params() definition regarding this
            % code.
            %{
            % make a copy and convert cvpartition generator to cvpartition
            % class instance here (this does the actual partitioning). We
            % do this because we (a) don't want to directly overwrite
            % what's in obj.ficlinearOpts and (b) can't pass generators to
            % our fitclinear invocation below, so we make a new variable
            % where we can replace the generator with a cvpartition object
            % and pass that instead.
            fitclinearOpts = obj.fitclinearOpts;
            if ~isempty(obj.CV_funhan)
                cvpart = obj.CV_funhan(X,Y);
                cv_idx = find(strcmp(fitclinearOpts,'CVPartition'));
                
                % do a sanity check. We get the function handle from the
                % 'CV' argument whenever check_lasso_params() is called, so
                % if we have one but not the other something very strange
                % is happening.
                assert(~isempty(cv_idx), 'CV function handle was found, but no ''CVPartition'' parameter was found in fitclinearOpts. Something is wrong.');
                fitclinearOpts{cv_idx+1} = cvpart;
            end
            
            Mdl = fitclinear(double(X),Y, fitclinearOpts{:});
            %}
            Mdl = fitclinear(double(X),Y, obj.fitclinearOpts{:});
            
            if isa(Mdl,'ClassificationPartitionedLinear')
                error('linearSvmClf does not support using fitclinear''s internal cross validation. Please wrap linearSvmClf in a crossValScore() object instead.');
            end
            
            obj.B = Mdl.Beta(:);
            obj.offset = Mdl.Bias;
            
            obj.prior = sum(obj.decisionFcn(Y) == 1)/length(Y);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
                
        function yfit_raw = score_samples(obj, X)
            yfit_raw = score_samples@linearModelEstimator(obj,X);
            yfit_raw = obj.scoreFcn(yfit_raw(:));
        end
        
        %% methods for dependent properties
        
        function obj = set.learner(obj, val)
            if ~strcmp(val, 'svm')
                error('Only svm Learners are supported by this function');
            end
            
            learner_idx = find(strcmp(obj.fitclinearOpts, 'Learner'));
            if isempty(learner_idx)
                obj.fitclinearOpts = [obj.fitclinearOpts, {'Learner', 'svm'}];
            else
                obj.fitclinearOpts{learner_idx + 1} = 'svm';
            end
        end
        
        function val = get.learner(obj)
            learner_idx = find(strcmp(obj.fitclinearOpts, 'Learner'));
            assert(~isempty(learner_idx), 'Learner not specified in fitclinearOpts. This is strange.');
            
            val = obj.fitclinearOpts{learner_idx+1};
        end
        
        function obj = set.intercept(obj, val)
            bias_idx = find(strcmp(obj.fitclinearOpts, 'FitBias'));
            if isempty(bias_idx)
                obj.fitclinearOpts = [obj.fitclinearOpts, {'FitBias', val}];
            else
                obj.fitclinearOpts{bias_idx + 1} = val;
            end
        end
        
        function val = get.intercept(obj)
            bias_idx = find(strcmp(obj.fitclinearOpts, 'FitBias'));
            assert(~isempty(bias_idx), 'FitBias not specified in fitclinearOpts. This is strange.');
            
            val = obj.fitclinearOpts{bias_idx+1};
        end
        
        function obj = set.lambda(obj, val)
            assert(val >= 0, 'lambda must be greater than 0');
            
            lambda_idx = find(strcmp(obj.fitclinearOpts,'Lambda'));
            if isempty(lambda_idx)
                obj.fitclinearOpts = [obj.fitclinearOpts, {'Lambda', val}];
            else
                obj.fitclinearOpts{lambda_idx + 1} = val;
            end
            
            obj.C = [];
        end
        
        function val = get.lambda(obj)            
            lambda_idx = find(strcmp(obj.fitclinearOpts, 'Lambda'));
            if ~any(lambda_idx)
                val = [];
            else
                val = obj.fitclinearOpts{lambda_idx+1};
            end
        end
        
        
        function obj = set.regularization(obj, val)
            % we cast to char() because bayesOpt will pass character
            % vectors in as type categorical(), which will cause
            % fitclinear to fail.
            val = char(val);
            assert(ismember(val,{'ridge','lasso','none'}), 'regularization must be ''ridge'', ''lasso'', or ''none''');
            
            regularization_idx = find(strcmp(obj.fitclinearOpts,'Regularization'));
                
            if strcmp(char(val),'none')
                if isempty(regularization_idx)
                    return
                else
                    obj.fitclinearOpts{regularization_idx:regularization_idx+1} = [];
                end
            else
                if isempty(regularization_idx)
                    obj.fitclinearOpts = [obj.fitclinearOpts, {'Regularization', val}];
                else
                    obj.fitclinearOpts{regularization_idx + 1} = val;
                end
            end
        end
        
        function val = get.regularization(obj)            
            regularization_idx = find(strcmp(obj.fitclinearOpts, 'Regularization'));
            if ~any(regularization_idx)
                val = [];
            else
                val = obj.fitclinearOpts{regularization_idx+1};
            end
        end
        
        function obj = set.scoreFcn(obj, val)
            if ischar(val)
                switch(val)
                    case 'doublelogit'
                        obj.scoreFcn0 = @(x1)(1/(1 + exp(-2*x1)));
                    case 'invlogit'
                        obj.scoreFcn0 = @(x1)(log(x1 / (1-x1)));
                    case 'ismax'
                        obj.scoreFcn0 = @(x1)(max(x1) == x1);
                    case 'logit'
                        obj.scoreFcn0 = @(x1)(1/(1+exp(-x1)));
                    case 'none'
                        obj.scoreFcn0 = @(x1)(x1);
                    case 'identity'
                        obj.scoreFcn0 = @(x1)(x1);
                    case 'sign'
                        obj.scoreFcn0 = @(x1)(sign(x1));
                    case 'symmetric'
                        obj.scoreFcn0 = @(x1)(2*x1 - 1);
                    case 'symetricismax'
                        obj.scoreFcn0 = @symmetricismax;
                    case 'symetriclogit'
                        obj.scoreFcn0 = @(x1)(2/(1 + exp(-x1)) - 1);
                    otherwise
                        error('scoreFcn %s not supported', val)
                end
            else            
                assert(isa(val, 'function_handle'), 'val must be a function_handle or a supported ScoreTransform option for fitclinear');
                obj.scoreFcn0 = val;
            end        
                                    
            % sync fitclinear args
            st_idx = find(strcmp(obj.fitclinearOpts, 'ScoreTransform'));
            if isempty(st_idx)
                obj.fitclinearOpts = [obj.fitclinearOpts, {'ScoreTransform', val}];
            else
                obj.fitclinearOpts{st_idx + 1} = val;
            end
        end
        
        function val = get.scoreFcn(obj)   
            val = obj.scoreFcn0;
        end
    end
    
    
    methods (Access = private)
        % if you let fitclinear perform cross-val internally to optimize
        % some parameter or other you will need to update fit to work on
        % ClassificationPartitionedLinear objects that are the result of
        % these CV routines, but it's not clear what purpose this would
        % serve so for now this is commented out.
        %{
        % fitclinear has an argument for specifying cross
        % validation folds, and this check incorporates a method for
        % allowing the user to specify function handles to cvpartition 
        % object generators instead of cvpartition instances. This is
        % useful if this modelEstimator ends up wrapped in some
        % crossValidator object, since passing a function handle then 
        % allows for cvpartition to be generated on demand based on the
        % particular fold slicing that's received from the crossValidator.
        function obj = check_cv_params(obj)     
            cv_idx = find(strcmp(obj.fitclinearOpts,'CVPartition'));
            if ~isempty(cv_idx)
                if isa(obj.fitclinearOpts{cv_idx+1},'function_handle')
                    obj.CV_funhan = obj.fitclinearOpts{cv_idx+1};
                end
            end
        end
        %}
    end
    
    methods (Access = private, Static)
        function val = symmetricismax(x1)
            val = max(x1) == x1;
            val(val == 0) = -1;
        end
    end
end
