% linearSvmRegressor a linearModelEstimator that fits a linear support
%   vector machine regression model to data (X,Y) using the C (slack)
%   parameterization by default and matlab's fitrlinear SVM implementation.
%
% estimator = linearSvmRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include intercept in univariate model. 
%                   Default: true;
%
%   'C'         - Slack parameter. To use Lambda specification leave this 
%                   unset and specify Lambda parameter in fitrlinearOpts. 
%                   Default C is used when neither is provided. Default: 1
%
%   'epsilon'   - Half width of epsilon intensive band. Can also be
%                   specified as fitrlinearOpts argument. See help
%                   fitrlinearOpts for default value.
%
%   'fitrlinearOpts'
%               - cell array of options to pass through to firlinear. See
%                   help fitrlinear for details. 
%                 Note: fitrlinear uses the Lambda parameterization. If 
%                   you specify a Lambda parameter here instead of a C 
%                   parameter in the linearSvmRegressor constructor then 
%                   that Lambda parameterization will be used. If you 
%                   specify both Lambda in fitrlinearOpts and C in the 
%                   linearSvmRegressor constructor then a warning will be 
%                   thrown and C will supercede Lambda.
%                 Note: 'CVPartition' can take either cvpartition objects
%                   or function handles to cvpartition generators that take
%                   (X,Y) as input and return a cvpartition. e.g.
%                   @(X,Y)cvpartition(ones(size(Y)),'KFOLD',5). 
%                   This can be helpful for on demand cvpartition
%                   generation, for instance if linearSvmRegressor is
%                   called by a crossValidator object, but you want
%                   something like lasso parameter estimation to be handled
%                   internally.
%                   
%
%   Note: both lambda and C are hyperparameters, but you should NOT try to
%   optimize both at the same time. Pick one parameterization only.
%
%   Note: fitrlinear has a bunch optimization routines built in. These 
%   have not yet been tested. It may be useful for lasso parameter
%   optimization in particular to take advantage of the LARS algorithm,
%   but otherwise its use is discouraged and you are encouraged to instead
%   wrap this routine in a bayesOptCV or gridSearchCV optimizer.
classdef linearSvmRegressor < linearModelEstimator & modelRegressor
    properties
        fitrlinearOpts = {};
        C = []; % defaults to 1 if neither C nor Lambda are found by the constructor
    end
    
    properties (Dependent)
        learner;
        intercept;
        epsilon;
    end
    
    properties (Dependent, Access = ?modelEstimator)
        lambda;
        regularization
    end  
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
        
        % CV_funhan = [];
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'intercept', 'C', 'lambda', 'epsilon', 'regularization'};
    end
          
    
    methods
        function obj = linearSvmRegressor(varargin)
            % defaults
            intercept = true;
            learner = 'svm';
            regularization = 'none';
            
            fitrlinearOpts_idx = find(strcmp(varargin,'fitrlinearOpts'));
            if ~isempty(fitrlinearOpts_idx)
                obj.fitrlinearOpts = varargin{fitrlinearOpts_idx + 1};
            end
            
            for i = 1:length(obj.fitrlinearOpts)
                if ischar(obj.fitrlinearOpts{i})
                    switch(obj.fitrlinearOpts{i})
                        case 'FitBias'
                            intercept = varargin{i+1};
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
            % we don't let fitrlinearOpts set these directly because
            % setting these in turn will modify fitclineearOpts, and who
            % knows what kind of strange behavior that feedback may cause
            % down the line. Better to have it in two separate invocations.
            obj.intercept = intercept;
            obj.learner = learner;
            obj.regularization = regularization;
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'intercept'
                            obj.intercept = varargin{i+1};
                        case 'C'
                            obj.C = varargin{i+1};                         
                            if any(strcmp(obj.fitrlinearOpts, 'Lambda'))
                                warning('Will override fitrlinearOpts Lambda = %0.3f, using C = %0.3f',obj.lambda, obj.C);
                            end
                        case 'epsilon'
                            if any(strcmp(obj.fitrlinearOpts, 'Epsilon'))
                                warning('Overriding firlinearOpts epsilon = %0.3f with linearSvmRegressor epsilon = 0.3%f',obj.epsilon, varargin{i+1});
                            end
                            obj.epsilon = varargin{i+1};
                        case 'regularization'
                            obj.regularization = varargin{i+1};
                        otherwise
                            warning('Option %s not supported', varargin{i});
                    end
                end
            end
            
            % this makes C=1 be the default if neither Lambda nor C are
            % specified.
            if ~any(strcmp(obj.fitrlinearOpts, 'Lambda'))
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
            
            % see note above check_cv_params definition below
            %{
            fitrlinearOpts = obj.fitrlinearOpts;
            if ~isempty(obj.CV_funhan)
                cvpart = obj.CV_funhan(X,Y);
                cv_idx = find(strcmp(fitrlinearOpts,'CVPartition'));
                
                % do a sanity check. We get the function handle from the
                % 'CV' argument whenever check_lasso_params() is called, so
                % if we have one but not the other something very strange
                % is happening.
                assert(~isempty(cv_idx), 'CV function handle was found, but no ''CVPartition'' parameter was found in fitrlinearOpts. Something is wrong.');
                fitrlinearOpts{cv_idx+1} = cvpart;
            end
            
            Mdl = fitrlinear(double(X),Y, fitrlinearOpts{:});
            %}
            Mdl = fitrlinear(double(X),Y, obj.fitrlinearOpts{:});
            
            if isa(Mdl,'ClassificationPartitionedLinear')
                error('linearSvmClf does not support using fitrlinear''s internal cross validation. Please wrap linearSvmClf in a crossValScore() object instead.');
            end
            
            obj.B = Mdl.Beta(:);
            obj.offset = Mdl.Bias;
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function obj = set.learner(obj, val)
            if ~strcmp(val, 'svm')
                error('Only svm Learners are supported by this function');
            end
            
            learner_idx = find(strcmp(obj.fitrlinearOpts, 'Learner'));
            if isempty(learner_idx)
                obj.fitrlinearOpts = [obj.fitrlinearOpts, {'Learner', 'svm'}];
            else
                obj.fitrlinearOpts{learner_idx + 1} = 'svm';
            end
        end
        
        function val = get.learner(obj)
            learner_idx = find(strcmp(obj.fitrlinearOpts, 'Learner'));
            assert(~isempty(learner_idx), 'Learner not specified in fitrlinearOpts. This is strange.');
            
            val = obj.fitrlinearOpts{learner_idx+1};
        end
        
        function obj = set.intercept(obj, val)
            bias_idx = find(strcmp(obj.fitrlinearOpts, 'FitBias'));
            if isempty(bias_idx)
                obj.fitrlinearOpts = [obj.fitrlinearOpts, {'FitBias', val}];
            else
                obj.fitrlinearOpts{bias_idx + 1} = val;
            end
        end
        
        function val = get.intercept(obj)
            bias_idx = find(strcmp(obj.fitrlinearOpts, 'FitBias'));
            assert(~isempty(bias_idx), 'FitBias not specified in fitrlinearOpts. This is strange.');
            
            val = obj.fitrlinearOpts{bias_idx+1};
        end
        
        function obj = set.lambda(obj, val)
            assert(val >= 0, 'lambda must be greater than 0');
            
            lambda_idx = find(strcmp(obj.fitrlinearOpts,'Lambda'));
            if isempty(lambda_idx)
                obj.fitrlinearOpts = [obj.fitrlinearOpts, {'Lambda', val}];
            else
                obj.fitrlinearOpts{lambda_idx + 1} = val;
            end
            
            % we can't know what C is without having a sample to fit to
            % (it depends on sample size), so we just erase this. If we're
            % setting lambda, we don't need C anyway.
            obj.C = [];
        end
        
        function val = get.lambda(obj)            
            lambda_idx = find(strcmp(obj.fitrlinearOpts, 'Lambda'));
            if ~any(lambda_idx)
                val = [];
            else
                val = obj.fitrlinearOpts{lambda_idx+1};
            end
        end
        
        function obj = set.epsilon(obj, val)
            assert(val >= 0, 'epsilon must be greater than 0');
            
            epsilon_idx = find(strcmp(obj.fitrlinearOpts,'Epsilon'));
            if isempty(epsilon_idx)
                obj.fitrlinearOpts = [obj.fitrlinearOpts, {'Epsilon', val}];
            else
                obj.fitrlinearOpts{epsilon_idx + 1} = val;
            end
        end
        
        function val = get.epsilon(obj)
            epsilon_idx = find(strcmp(obj.fitrlinearOpts, 'Epsilon'));
            if ~any(epsilon_idx)
                val = [];
            else
                val = obj.fitrlinearOpts{epsilon_idx+1};
            end
        end
        
        function obj = set.regularization(obj, val)
            % we cast to char() because bayesOpt will pass character
            % vectors in as type categorical(), which will cause
            % fitrlinear to fail.
            val = char(val);
            assert(ismember(val,{'ridge','lasso','none'}), 'regularization must be ''ridge'', ''lasso'', or ''none''');
            
            regularization_idx = find(strcmp(obj.fitrlinearOpts,'Regularization'));
                
            if strcmp(val,'none')
                if isempty(regularization_idx)
                    return
                else
                    obj.fitrlinearOpts{regularization_idx:regularization_idx+1} = [];
                end
            else
                % we cast to char() because bayesOpt will pass character
                % vectors in as type categorical(), which will cause
                % fitrlinear to fail.
                if isempty(regularization_idx)
                    obj.fitrlinearOpts = [obj.fitrlinearOpts, {'Regularization', val)];
                else
                    obj.fitrlinearOpts{regularization_idx + 1} = val;
                end
            end
        end
        
        function val = get.regularization(obj)            
            regularization_idx = find(strcmp(obj.fitrlinearOpts, 'Regularization'));
            if ~any(regularization_idx)
                val = [];
            else
                val = obj.fitrlinearOpts{regularization_idx+1};
            end
        end
    end
    
    methods (Access = private)
        % if you let fitrlinear perform cross-val internally to optimize
        % some parameter or other you will need to update fit to work on
        % ClassificationPartitionedLinear objects that are the result of
        % these CV routines, but it's not clear what purpose this would
        % serve so for now this is commented out.
        %{
        % fitrlinear has an argument for specifying cross
        % validation folds, and this check incorporates a method for
        % allowing the user to specify function handles to cvpartition 
        % object generators instead of cvpartition instances. This is
        % useful if this modelEstimator ends up wrapped in some
        % crossValidator object, since passing a function handle then 
        % allows for cvpartition to be generated on demand based on the
        % particular fold slicing that's received from the crossValidator.
        function obj = check_cv_params(obj)     
            cv_idx = find(strcmp(obj.fitrlinearOpts,'CVPartition'));
            if ~isempty(cv_idx)
                if isa(obj.fitrlinearOpts{cv_idx+1},'function_handle')
                    obj.CV_funhan = obj.fitrlinearOpts{cv_idx+1};
                end
            end
        end
        %}
    end
end
