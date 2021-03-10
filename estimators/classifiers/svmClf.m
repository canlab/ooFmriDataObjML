% svmClf a linearModelEstimator that fits a linear support
%   vector machine classification model to data (X,Y) using the C (slack)
%   parameterization by default and matlab's fitcsvm SVM implementation.
%
% estimator = linearSvmRegressor([options])
%
%   options ::
%
%
%   'fitcsvmOpts'
%               - cell array of options to pass through to firlinear. See
%                   help fitrlinear for details. 
%                 Note: fitcsvm uses the Lambda parameterization. If 
%                   you specify a Lambda parameter here instead of a C 
%                   parameter in the svmClf constructor then 
%                   that Lambda parameterization will be used. If you 
%                   specify both Lambda in fitcsvmOpts and C in the 
%                   svmClf constructor then a warning will be 
%                   thrown and C will supercede Lambda.
%                 Note: fitcsvmOpts has the option of performing
%                   hyperparameter optimization internally via kfold, hold
%                   out or custom cvpartition specification. This is not
%                   supported via svmClf. Wrap svmClf in a
%                   hyperparameter optimization object like bayesOptCV
%                   instead.
%                   
%
%   Note: both lambda and C are hyperparameters, but you should NOT try to
%   optimize both at the same time. Pick one parameterization only.
%
%   Note: fitcsvm has a bunch optimization routines built in. These 
%   have not yet been tested. It may be useful for lasso parameter
%   optimization in particular to take advantage of the LARS algorithm,
%   but otherwise its use is discouraged and you are encouraged to instead
%   wrap this routine in a bayesOptCV or gridSearchCV optimizer.
classdef svmClf < modelClf
    properties
        fitcsvmOpts = {};
        C = [];
    end
    
    properties (Dependent)
        scoreFcn;
    end
    
    properties (Dependent, SetAccess = ?baseEstimator)
        kernel;
        kernelScale;
        order;
        nu;
    end  
    
    properties (Access = private, Hidden)
        % we need to store scoreFcn in both the fitcsvm format (which is
        % either a function handle or a string) and in a pure function
        % handle format. To avoid confusion we keep the pure function
        % handle format hidden from the user. This should never be accessed
        % directly though. Always call by invoking scoreFcn, which will use
        % it's get method to pull this data.
        scoreFcn0 = @(x1)(x1);
    end
    
    properties (SetAccess = protected)                
        isFitted = false;
        fitTime = -1;
        
        Mdl = [];
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'kernel', 'kernelScale', 'order', 'nu'};
    end
          
    properties (Dependent = true, SetAccess = protected)
        classLabels;
        offset;
    end
    
    methods
        %% constructor
        function obj = svmClf(varargin)
            % defaults
            
            scoreFcn = 'none';
            
            fitcsvmOpts_idx = find(strcmp(varargin,'fitcsvmOpts'));
            if ~isempty(fitcsvmOpts_idx)
                obj.fitcsvmOpts = varargin{fitcsvmOpts_idx + 1};
            end
            
            for i = 1:length(obj.fitcsvmOpts)
                if ischar(obj.fitcsvmOpts{i})
                    switch(obj.fitcsvmOpts{i})
                        case 'ScoreTransform'
                            scoreFcn = obj.fitcsvmOpts{i + 1};
                        case 'KFold'
                            error('Internal cross validation is not supported. Please wrap svmClf in a bayesOptCV object or similar instead');
                        case 'CVPartition'
                            error('Internal cross validation is not supported. Please wrap svmClf in a bayesOptCV object or similar instead');
                        case 'Holdout'
                            error('Internal cross validation is not supported. Please wrap svmClf in a bayesOptCV object or similar instead');
                        case 'CrossVal'
                            error('Internal cross validation is not supported. Please wrap svmClf in a bayesOptCV object or similar instead');
                    end
                end
            end
            % we don't let fitcsvmOpts set these directly because
            % setting these in turn will modify fitclineearOpts, and who
            % knows what kind of strange behavior that feedback may cause
            % down the line. Better to have it in two separate invocations.
            obj.scoreFcn = scoreFcn;
            
            % by parsing fitcsvmOpts first we give override priority to
            % arguments passed directly to svmClf, which we parse now
            % and will overwrite anything we did earlier if it differs.
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'C'
                            obj.C = varargin{i+1};                         
                            if any(strcmp(obj.fitcsvmOpts, 'Lambda'))
                                warning('Will override fitcsvmOpts Lambda = %0.3f, using C = %0.3f',obj.lambda, obj.C);
                            end
                        case 'kernel'
                            obj.kernel = varargin{i+1};
                            varargin{i+1} = [];
                        case 'nu'
                            obj.nu = varargin{i+1};
                        case 'kernelScale'
                            obj.scale = varargin{i+1};
                        case 'order'
                            obj.order = varargin{i+1};
                        case 'fitcsvmOpts'
                            continue;
                        otherwise
                            warning('Option %s not supported', varargin{i});
                    end
                end
            end
            
            if ~any(strcmp(obj.fitcsvmOpts, 'Lambda'))
                obj.C = 1;
            end
            
            % see note above check_cv_params definition below
            % obj.check_cv_params();
        end
        
        %% fit method
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            
            
            obj.Mdl = fitcsvm(double(X),Y, obj.fitcsvmOpts{:});
            
            if isa(obj.Mdl,'ClassificationPartitionedLinear')
                error('svmClf does not support using fitcsvm''s internal cross validation. Please wrap svmClf in a crossValScore() object instead.');
            end
            
            obj.prior = sum(obj.decisionFcn(Y) == 1)/length(Y);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
                
        %% methods for prediction
        
        function yfit_raw = score_samples(obj, X, varargin)
            [~,yfit_raw] = obj.Mdl.predict(double(X));
            yfit_raw = yfit_raw(:,obj.Mdl.ClassNames == 1);
            if any(isnan(yfit_raw))
                warning('SVM model is returning nans. Check output, and don''t trust optimization algs');
            end
        end        
         
        function yfit_raw = score_null(obj, n)
            yfit_raw = repmat(obj.scoreFcn(obj.offset),n,1); 
            
            st_idx = find(strcmp(obj.fitcsvmOpts, 'ScoreTransform'));
            if ~(strcmp(obj.fitcsvmOpts{st_idx+1},'none') || strcmp(obj.fitcsvmOpts{st_idx+1}, 'identity'))
                warning('svmClf.score_null() behavior has not been validated with non-trivial scoreFcn. Please check the results.');
            end
        end
        
        function yfit = predict(obj, X)
            yfit = obj.Mdl.predict(double(X));
            yfit(yfit == 0) = -1;
        end
        
        function d = decisionFcn(~, scores)
            d = zeros(size(scores));
            d(scores > 0) = 1;
            d(scores <=0) = -1;
        end
        
        %% methods for dependent properties   
        
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
        
        function set.nu(obj, val)
            assert(val >= 0, 'lambda must be greater than 0');
            
            nu_idx = find(strcmp(obj.fitcsvmOpts,'Nu'));
            if isempty(nu_idx)
                obj.fitcsvmOpts = [obj.fitcsvmOpts, {'Nu', val}];
            else
                obj.fitcsvmOpts{nu_idx + 1} = val;
            end
            
            obj.C = [];
        end
        
        function val = get.nu(obj)            
            nu_idx = find(strcmp(obj.fitcsvmOpts, 'Nu'));
            if ~any(nu_idx)
                val = [];
            else
                val = obj.fitcsvmOpts{nu_idx+1};
            end
        end
        
        function set.kernel(obj, val)
            
            k_idx = find(strcmp(obj.fitcsvmOpts, 'KernelFunction'));
            if isempty(k_idx)
                obj.fitcsvmOpts{end+1} = 'KernelFunction';
                k_idx = length(obj.fitcsvmOpts);
            end
            
            if isa(val,'function_handle')
                obj.fitcsvmOpts{k_idx + 1} = val;
            else
                val = char(val);
                if ismember(val, {'linear', 'gaussian', 'rbf', 'polynomial'})
                    obj.fitcsvmOpts{k_idx + 1} = val;
                else
                    error('%s kernel type is unsupported', val);
                end
            end 
        end
        
        function val = get.kernel(obj)
            k_idx = find(strcmp(obj.fitcsvmOpts, 'KernelFunction'));
            if isempty(k_idx)
                val = 'linear';
            else
                val = obj.fitcsvmOpts{k_idx + 1};
            end
        end
        
        function set.kernelScale(obj, val)
            k_idx = find(strcmp(obj.fitcsvmOpts, 'KernelScale'));
            if isempty(k_idx)
                obj.fitcsvmOpts{end+1} = 'KernelScale';
                k_idx = length(obj.fitcsvmOpts);
            end
            
            if isnumeric(val)
                obj.fitcsvmOpts{k_idx} = val;
            elseif strcmp(val, 'auto')
                obj.fitcsvmOpts{k_idx} = val;
            else
                error('Unsupported kernel scale supplied');
            end
        end
        
        function val = get.kernelScale(obj)
            k_idx = find(strcmp(obj.fitcsvmOpts, 'KernelScale'));
            if isempty(k_idx)
                val = 1;
            else
                val = obj.fitcsvmOpts{k_idx + 1};
            end
        end
        
        
        function set.order(obj, val)
            
            p_idx = find(strcmp(obj.fitcsvmOpts, 'PolynomialOrder'));
            if isempty(p_idx) && val > 0
                obj.fitcsvmOpts{end+1} = 'PolynomialOrder';
                p_idx = length(obj.fitcsvmOpts);
            end
            
            if val > 0
                obj.fitcsvmOpts{p_idx+1} = val;
            else
                error('Unsupported polynomial order supplied');
            end
        end
        
        function val = get.order(obj)
            p_idx = find(strcmp(obj.fitcsvmOpts, 'PolynomialOrder'));
            if isempty(p_idx)
                val = 3; % default value
            else
                val = obj.fitcsvmOpts{p_idx + 1};
            end
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
                assert(isa(val, 'function_handle'), 'val must be a function_handle or a supported ScoreTransform option for fitcsvm');
                obj.scoreFcn0 = val;
            end        
                                    
            % sync fitcsvm args
            st_idx = find(strcmp(obj.fitcsvmOpts, 'ScoreTransform'));
            if isempty(st_idx)
                obj.fitcsvmOpts = [obj.fitcsvmOpts, {'ScoreTransform', val}];
            else
                obj.fitcsvmOpts{st_idx + 1} = val;
            end
        end
        
        function val = get.scoreFcn(obj)   
            val = obj.scoreFcn0;
        end
        
        
        function val = get.offset(obj)
            if isempty(obj.Mdl)
                val = 0;
            else
                val = obj.Mdl.Bias;
            end
        end
        
        function set.offset(~, ~)
            warning('You shouldn''t be setting offset directly. offset is part of obj.Mdl. Doing nothing.');
        end
    end
    
    
    methods (Access = private, Static)
        function val = symmetricismax(x1)
            val = max(x1) == x1;
            val(val == 0) = -1;
        end
    end
end
