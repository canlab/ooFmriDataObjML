% linearSvmClf a linearModelEstimator that fits a linear support
%   vector machine classification model to data (X,Y) using the C (slack)
%   parameterization by default and matlab's fitcecoc SVM implementation.
%
% estimator = linearSvmRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include intercept in univariate model. 
%                   Default: true;
%
%   'C'         - Slack parameter. To use Lambda specification leave this 
%                   unset and specify Lambda parameter in fitcecocOpts. 
%                   Default C is used when neither is provided. Default: 1
%
%   'epsilon'   - Half width of epsilon intensive band. Can also be
%                   specified as fitcecocOpts argument. See help
%                   fitcecocOpts for default value.
%
%   'regularization'
%               - 'ridge', 'lasso', or 'none'
%
%   'fitcecocOpts'
%               - cell array of options to pass through to firlinear. See
%                   help fitrlinear for details. 
%                 Note: fitcecoc uses the Lambda parameterization. If 
%                   you specify a Lambda parameter here instead of a C 
%                   parameter in the linearSvmClf constructor then 
%                   that Lambda parameterization will be used. If you 
%                   specify both Lambda in fitcecocOpts and C in the 
%                   linearSvmClf constructor then a warning will be 
%                   thrown and C will supercede Lambda.
%                 Note: fitcecocOpts has the option of performing
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
%   Note: fitcecoc has a bunch optimization routines built in. These 
%   have not yet been tested. It may be useful for lasso parameter
%   optimization in particular to take advantage of the LARS algorithm,
%   but otherwise its use is discouraged and you are encouraged to instead
%   wrap this routine in a bayesOptCV or gridSearchCV optimizer.
classdef multiclassLinearSvmClf < linearModelEstimator & modelClf
    properties
        fitcecocOpts = {};
    end
    
    properties (Dependent)
        scoreFcn;
    end
    
    properties (Dependent, SetAccess = ?baseEstimator)
        regularization
        lambda
        intercept
    end
    
    properties (Dependent, SetAccess = private)        
        learner;
        
        B;
        offset;
        
        codingDesign;
        classLabels;
    end
    
    properties (Access = private, Hidden)
        % we need to store scoreFcn in both the fitcecoc format (which is
        % either a function handle or a string) and in a pure function
        % handle format. To avoid confusion we keep the pure function
        % handle format hidden from the user. This should never be accessed
        % directly though. Always call by invoking scoreFcn, which will use
        % it's get method to pull this data.
        scoreFcn0 = @(x1)(x1);
    end
    
    properties (SetAccess = private)         
        NClasses = 2;
        nClf = 1;
        
        isFitted = false;
        fitTime = -1;
        
        % CV_funhan = [];
        
        Mdl = [];
        learnerTemplates = {};
        
        lossFcn = @(y, yfit)(max(0, 1 - y .* yfit)./2)
    end
    
    properties (Access = ?Estimator)
        % these are the default, but they will be extended by the
        % constructor for multiclass classification to include
        % hyperparameters for each individual classifier
        hyper_params = {'intercept', 'lambda', 'regularization'};
    end
          
    
    methods
        % we design this constructor so you can pass in individual
        % hyperparameters to individual classifiers via the constructor
        % arguments. Individual classifiers are indicated via underscores.
        % If no underscore is provided, setting is proprogated to all
        % classifiers.
        function obj = multiclassLinearSvmClf(varargin)            
            %%
            % pull fitcecoc options
            fitcecocOpts_idx = find(strcmp(varargin,'fitcecocOpts'));
            if ~isempty(fitcecocOpts_idx)
                obj.fitcecocOpts = varargin{fitcecocOpts_idx + 1};
            end
            
            %%
            % figure out how many hyperparameters we need based on NClasses
            % and Coding Design.
            if any(strcmp(varargin,'NClasses'))
                obj.NClasses = varargin{find(strcmp(varargin,'NClasses')) + 1};
            else
                warning('NClasses not specified, using default (NClasses = 2).');
                obj.NClasses = 2;
            end
            
            codingDesign = 'onevsall'; % default
            if any(strcmp(obj.fitcecocOpts,'Coding'))
                c_idx = find(strcmp(obj.fitcecocOpts,'Coding'));
                codingDesign = obj.fitcecocOpts(c_idx + 1);
                
                if ~(strcmp(obj.codingDesign, 'onevsall') || strcmp(obj.codingDesign, 'onevsone'))
                    error('Coding scheme must be ''onevsall'' or ''onevsall'', but found %s', obj.codingDesign);
                end
            end 
            obj.codingDesign = codingDesign;
            if obj.NClasses > 2
                switch(obj.codingDesign)
                    case 'onevsall'
                        if obj.NClasses == 2
                            obj.nClf = 1;
                        else
                            obj.nClf = obj.NClasses;
                        end
                    case 'onevsone' % not implemented yet.
                        obj.nClf = obj.NClasses*(obj.NClasses - 1);
                    otherwise
                        error('Unsupported coding scheme');
                end
            else
                obj.nClf = 1; % if there are only 2 classes,  we only need 1 clf
            end
            
            %%
            % set multiclass classifier (the same for all constituent
            % binary classifiers) defaults
            scoreFcn = 'none';
            intercept = true;
            learner = 'svm';
            regularization = 'none';
            obj.learnerTemplates = cell(obj.nClf,1);
            for i = 1:obj.nClf
                % this is what makes it a fitcecoc backend instead of a
                % fitcsvm backend I believe. Modify if making a nonlinear
                % version of this class
                obj.learnerTemplates{i} = templateLinear();
            end
            
            %%
            % Initialize hyperparameters and set defaults for constituent 
            % binary classifier
            % create 'dynamic' properties, one for each classifier we need.
            mdl_params = {'intercept', 'lambda', 'regularization'};
            for i = 1:obj.nClf
                for s = 1:length(mdl_params)
                    propname = [mdl_params{s}, '_', int2str(i)];
                    
                    obj.addDynProp(propname);
                    this_prop = findprop(obj,propname);
                    this_prop.SetAccess = 'protected'; % this forces us to overload get_params and set_params here
                    this_prop.NonCopyable = false;
                    % we define get and set methods like so (for
                    % example, using lambda_1 as the regularization
                    % parameter for classifier 1),
                    %   this_prop.SetMethod = @(obj, val)(set_lambda(obj, val, 1));
                    this_prop.SetMethod = eval(['@(obj,val)(set_' mdl_params{s} '(obj, val, ' int2str(i) '));']);
                    this_prop.GetMethod = eval(['@(obj)(get_' mdl_params{s} '(obj, ' int2str(i) '));']);

                    obj.hyper_params = [obj.hyper_params, propname];

                    % define binary classifier defaults,
                    switch(mdl_params{s})
                        case 'intercept'
                            eval([propname, ' = true;']);
                        case 'regularization'
                            eval([propname, ' = ''none'';']);
                    end
                end
            end
            
            %% parse fitcecocOpts
            % we do this before parsing multiclassLinearSvmClf args so that
            % anything that's specified otherwise there will overwrite
            % these and take precedence.
            fitcecocOpts = obj.fitcecocOpts;
            for i = 1:length(fitcecocOpts)
                if ischar(fitcecocOpts{i})
                    switch(fitcecocOpts{i})
                        case 'ScoreTransform'
                            scoreFcn = fitcecocOpts{i + 1};
                        case 'FitBias'
                            intercept = logical(fitcecocOpts{i+1});
                        case 'Learner'
                            learner = fitcecocOpts{i+1};
                        case 'Regularization'
                            regularization = fitcecocOpts{i+1};
                        case 'Coding'
                            fitcecocOpts{i+1} = [];
                        case 'Cost'
                            error('Cost option to fitcecoc is not implemented in multiclassLinearSvmClf');
                        case 'Learners'
                            error('Learners options in fitcecoc are not currently implemented for multiclassLinearSvmClf objects.');
                        case 'KFold'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                        case 'CVPartition'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                        case 'Holdout'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                        case 'CrossVal'
                            error('Internal cross validation is not supported. Please wrap linearSvmClf in a bayesOptCV object or similar instead');
                    end
                end
            end
            % we don't let fitcecocOpts set these directly because
            % setting these in turn will modify fitclineearOpts, and who
            % knows what kind of strange behavior that feedback may cause
            % down the line. Better to have it in two separate invocations.
            obj.scoreFcn = scoreFcn;
            obj.intercept = intercept;
            obj.learner = learner;
            obj.regularization = regularization;
            
            % by parsing fitcecocOpts first we give override priority to
            % arguments passed directly to linearSvmClf, which we parse now
            % and will overwrite anything we did earlier if it differs.
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'NClasses'
                            continue; % already handled
                        case 'C'
                            warning('Only Lambda classification is supported by multiclassLinearSVMClf');
                        otherwise
                            if ismember(varargin{i}, obj.hyper_params)
                                obj.(varargin{i}) = varargin{i+1};
                                varargin{i+1} = [];
                            else
                                warning('Option %s not supported. Ignoring.', varargin{i});
                            end
                    end
                end
            end
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            if ~iscategorical(Y)
                warning('Expected categorical class labels but recieved %s. Attempting naive conversion.', class(Y));
            end
                                    
            for i = 1:obj.nClf
                obj.Mdl = fitcecoc(double(X), Y, obj.fitcecocOpts{:});

                if isa(obj.Mdl,'ClassificationPartitionedLinear')
                    error('multiclassLinearSvmClf does not support using fitcecoc''s internal cross validation. Please wrap multiclassLinearSvmClf in a crossValScore() object instead.');
                end

                obj.prior = zeros(size(obj.classLabels));
                for i = 1:length(obj.classLabels)
                    obj.prior(i) = sum(categorical(Y) == categorical(obj.classLabels(i)))/length(Y);
                end
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
                
        function yfit_raw = score_samples(obj, X, varargin)
            yfit_raw = score_samples@linearModelEstimator(obj,X);
            yfit_raw = obj.scoreFcn(yfit_raw);
        end        
         
        function yfit_raw = score_null(obj, n)
            yfit_raw = score_null@linearModelEstimator(obj,n);
            yfit_raw = obj.scoreFcn(yfit_raw);
            
            st_idx = find(strcmp(obj.fitcecocOpts, 'ScoreTransform'));
            if ~(strcmp(obj.fitcecocOpts{st_idx+1},'none') || strcmp(obj.fitcecocOpts{st_idx+1}, 'identity'))
                warning('linearSvmClf.score_null() behavior has not been validated with non-trivial scoreFcn. Please check the results.');
            end
        end
        
        
        function labels = decisionFcn(obj, scores)
            K = size(obj.Mdl.CodingMatrix,1);
            J = size(obj.Mdl.CodingMatrix,2);
            assert(J == obj.nClf,'Looks like NClasses was misspecified or the decisionFcn definition was misunderstood... Please review https://www.mathworks.com/help/stats/classificationecoc.predict.html#bufel6__sep_sharedBinaryLoss and fix this');
            
            n = size(scores,1);
            loss = zeros(n,K);
            for k = 1:K
                tmp = zeros(n, 1);
                weighting = 0;
                for j = 1:obj.nClf
                    this_y = obj.Mdl.CodingMatrix(k,j)*ones(n,1);
                    this_yfit = scores(:,j);
                    tmp = tmp + abs(obj.Mdl.CodingMatrix(k,j)).*obj.lossFcn(this_y, this_yfit);
                    weighting = weighting + abs(obj.Mdl.CodingMatrix(k,j));
                end
                loss(:,k) = tmp./weighting;
            end
            
            [a,b] = find(loss == min(loss,[],2));
            [~,I] = sort(a);
            labels = obj.classLabels(b(I));            
        end
        
        %% methods for dependent properties
        
        function val = get.B(obj)
            val = [];
            if ~isempty(obj.Mdl)
                val = cell2mat(cellfun(@(x1)(x1.Beta(:)), obj.Mdl.BinaryLearners, 'UniformOutput', false)');
            end
        end
        
        function set.B(obj, ~)
            warning('You shouldn''t be setting B directly. B is part of obj.Mdl. Doing nothing.');
        end
        
        function val = get.offset(obj)
            if isempty(obj.Mdl)
                val = 0;
            else
                val = cell2mat(cellfun(@(x1)(x1.Bias(:)), obj.Mdl.BinaryLearners, 'UniformOutput', false));
            end
            val = val(:)';
        end
        
        function set.offset(obj, ~)
            warning('You shouldn''t be setting offset directly. offset is part of obj.Mdl. Doing nothing.');
        end
        
        function val = get.classLabels(obj)
            if isempty(obj.Mdl)
                val = [];
            else
                val = obj.Mdl.ClassNames;
            end
        end
        
        function set.classLabels(~, ~)
            error('You shouldn''t be setting classLabels directly. This is set automatically when calling fit.');
        end
        
        function val = get.codingDesign(obj)
            c_idx = find(strcmp(obj.fitcecocOpts,'Coding'));
            if isempty(c_idx)
                val = [];
            else
                val = obj.fitcecocOpts{c_idx + 1};
            end
        end
        
        function set.codingDesign(obj, val)
            val = char(val);
            
            c_idx = find(strcmp(obj.fitcecocOpts,'Coding'));
            if isempty(c_idx)
                obj.fitcecocOpts{end+1} = 'Coding';
                c_idx = length(obj.fitcecocOpts) + 1;
            end
            obj.fitcecocOpts{c_idx} = val;
        end
        
        function set.scoreFcn(obj, val)
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
                assert(isa(val, 'function_handle'), 'val must be a function_handle or a supported ScoreTransform option for fitcecoc');
                obj.scoreFcn0 = val;
            end        
                                    
            % sync fitcecoc args
            st_idx = find(strcmp(obj.fitcecocOpts, 'ScoreTransform'));
            if isempty(st_idx)
                obj.fitcecocOpts = [obj.fitcecocOpts, {'ScoreTransform', val}];
            else
                obj.fitcecocOpts{st_idx + 1} = val;
            end
        end
        
        function val = get.scoreFcn(obj)   
            val = obj.scoreFcn0;
        end
        
        %% Dependent meta property methods
        % These properties are not classifier specific but set and return
        % values from all constituent classifiers. They can be useful if
        % you want to optimize hyperparameters in a homogeneous way (to
        % reduce the dimensionality of the search space for instance)
        
        function set.learner(obj, val)
            for i = 1:obj.nClf
                set_learner(obj, val, i);
            end
        end
        
        function val = get.learner(obj)
            val = cell(1, obj.nClf);
            for i = 1:obj.nClf
                val{i} = get_learner(obj, i);
            end
        end
        
        function set.intercept(obj, val)
            for i = 1:obj.nClf
                set_intercept(obj, val, i);
            end
        end
        
        function val = get.intercept(obj)
            val = cell(1, obj.nClf);
            for i = 1:obj.nClf
                val{i} = get_intercept(obj, i);
            end
        end
        
        function set.regularization(obj, val)
            for i = 1:obj.nClf
                set_regularization(obj, val, i);
            end
        end
        
        function val = get.regularization(obj)
            val = cell(1, obj.nClf);
            for i = 1:obj.nClf
                val{i} = get_regularization(obj, i);
            end
        end
        
        function set.lambda(obj, val)
            for i = 1:obj.nClf
                set_lambda(obj, val, i);
            end
        end
        
        function val = get.lambda(obj)
            val = cell(1, obj.nClf);
            for i = 1:obj.nClf
                    val{i} = get_lambda(obj, i);
            end
        end
        
        %% Dependent property proto-methods
        % These are not actually dependent property set/get 
        % methods, but they define functions which will be converted to 
        % depenent properties and passed as function handles to dynamic 
        % properties in our constructor. These proto-dependent methods are
        % indicated by get_ or set_ where a true dependent method would be
        % get. or set. (period not underscore). Additionally they take a
        % classifier index as input which will be statically defined before
        % being passed to dynamic properties in the constructor, thereby 
        % creating a dependent property method that applies to the
        % properties of a particular classifier.
        
        function set_learner(obj, val, clfIdx)
            val = char(val);
            obj.learnerTemplates{clfIdx}.ModelParams.Learner = val;
        end
        
        function val = get_learner(obj, clfIdx)
            val = obj.learnerTemplates{clfIdx}.ModelParams.Learner;
        end
        
        function set_intercept(obj, val, clfIdx)
            val = logical(val);
            obj.learnerTemplates{clfIdx}.ModelParams.FitBias = val;
        end
        
        function val = get_intercept(obj, clfIdx)
            val = obj.learnerTemplates{clfIdx}.ModelParams.FitBias;
        end
        
        function set_lambda(obj, val, clfIdx)
            obj.learnerTemplates{clfIdx}.ModelParams.Lambda = val;
        end
        
        function val = get_lambda(obj, clfIdx)  
            val = obj.learnerTemplates{clfIdx}.ModelParams.Lambda;
        end
        
        function set_regularization(obj, val, clfIdx)           
            % we cast to char() because bayesOpt will pass character
            % vectors in as type categorical(), which will cause
            % fitcecoc to fail.
            val = char(val);
            assert(ismember(val,{'ridge','lasso','none'}), 'regularization must be ''ridge'', ''lasso'', or ''none''');
                
            if strcmp(val,'none')
                obj.learnerTemplates{clfIdx}.ModelParams.Regularization = [];
            else
                obj.learnerTemplates{clfIdx}.ModelParams.Regularization = val;
            end
        end
        
        function val = get_regularization(obj, clfIdx)   
            val = obj.learnerTemplates{clfIdx}.ModelParams.Regularization;
        end
        
        
    end
    
    
    methods (Access = private, Static)
        function val = symmetricismax(x1)
            val = max(x1) == x1;
            val(val == 0) = -1;
        end
    end
end
