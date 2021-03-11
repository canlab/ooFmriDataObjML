% svmRegressor an estimato that fits a support vector machine regression
%   model, including kernel based models, to data (X,Y) using lambda
%   vector machine regression model to data (X,Y) using epsilon
%   parameterization by default and matlab's fitrsvm SVM implementation.
%   Best for low dimensional problems.
%
% estimator = svmRegressor([options])
%
%   options ::
%
%   'intercept' - true/false. Include intercept in univariate model. 
%                   Default: true;
%
%   'epsilon'   - Half width of epsilon intensive band. Can also be
%                   specified as fitrsvmOpts argument. See help
%                   fitrsvmOpts for default value.
%
%   'fitrsvmOpts'
%               - cell array of options to pass through to firsvm. See
%                   help fitrsvm for details. 
%                 Note: 'CVPartition' can take either cvpartition objects
%                   or function handles to cvpartition generators that take
%                   (X,Y) as input and return a cvpartition. e.g.
%                   @(X,Y)cvpartition(ones(size(Y)),'KFOLD',5). 
%                   This can be helpful for on demand cvpartition
%                   generation, for instance if svmRegressor is
%                   called by a crossValidator object, but you want
%                   something like lasso parameter estimation to be handled
%                   internally.
%
%   Note: fitrsvm has a bunch optimization routines built in. These 
%   have not yet been tested. It may be useful for lasso parameter
%   optimization in particular to take advantage of the LARS algorithm, if
%   that's even built into fitrsvm, and I don't know if it is,
%   but otherwise its use is discouraged and you are encouraged to instead
%   wrap this routine in a bayesOptCV or gridSearchCV optimizer.
classdef svmRegressor < modelRegressor
    properties
        fitrsvmOpts = {};
    end
    
    properties (Dependent, SetAccess = ?baseEstimator)
        epsilon;
        kernel;
        scale;
        order;
    end  
    
    properties (SetAccess = protected)                
        isFitted = false;
        fitTime = -1;
        
        Mdl = [];

        offset_null = 0;
        
        % CV_funhan = [];
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {'kernel', 'kernelScale', 'order', 'epsilon'};
    end
          
    
    methods
        function obj = svmRegressor(varargin)
            % defaults
            kernel = 'linear';
            order = [];
            scale = 'auto';
            epsilon = [];
            
            fitrsvmOpts_idx = find(strcmp(varargin,'fitrsvmOpts'));
            if ~isempty(fitrsvmOpts_idx)
                obj.fitrsvmOpts = varargin{fitrsvmOpts_idx + 1};
            end
            
            for i = 1:length(obj.fitrsvmOpts)
                if ischar(obj.fitrsvmOpts{i})
                    switch(obj.fitrsvmOpts{i})
                        case 'KernelScale'
                            scale = varargin{i+1};
                        case 'KernelFunction'
                            kernel = varargin{i+1};
                        case 'Epsilon'
                            epsilon = varargin{i+1};
                        case 'PolynomialOrder'
                            order = varargin{i+1};
                        case 'KFold'
                            error('Internal cross validation is not supported. Please wrap svmRegressor in a bayesOptCV object or similar instead');
                        case 'CVPartition'
                            error('Internal cross validation is not supported. Please wrap svmRegressor in a bayesOptCV object or similar instead');
                        case 'Holdout'
                            error('Internal cross validation is not supported. Please wrap svmRegressor in a bayesOptCV object or similar instead');
                        case 'CrossVal'
                            error('Internal cross validation is not supported. Please wrap svmRegressor in a bayesOptCV object or similar instead');
                    end
                end
            end
            % we don't let fitrsvmOpts set these directly because
            % setting these in turn will modify fitrsvmOpts, and who
            % knows what kind of strange behavior that feedback may cause
            % down the line. Better to have it in two separate invocations.
            obj.kernel = kernel;
            obj.order = order;
            obj.scale = scale;
            obj.epsilon = epsilon;
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'epsilon'
                            if any(strcmp(obj.fitrsvmOpts, 'Epsilon'))
                                warning('Overriding fitrsvmOpts epsilon = %0.3f with svmRegressor epsilon = 0.3%f',obj.epsilon, varargin{i+1});
                            end
                            obj.epsilon = varargin{i+1};
                            varargin{i+1} = [];
                        case 'kernel'
                            if any(strcmp(obj.fitrsvmOpts, 'KernelFunction'))
                                warning('Overriding fitrsvmOpts kernel = %s with svmRegressor kernel = %s',obj.kernel, varargin{i+1});
                            end
                            obj.kernel = varargin{i+1};
                            varargin{i+1} = [];
                        case 'scale'
                            if any(strcmp(obj.fitrsvmOpts, 'KernelScale'))
                                warning('Overriding fitrsvmOpts kernel scale = %0.3f with svmRegressor kernel scale = %0.3f',obj.scale, varargin{i+1});
                            end
                            obj.scale = varargin{i+1};
                            varargin{i+1} = [];
                        case 'order'
                            if any(strcmp(obj.fitrsvmOpts, 'PolynomialOrder'))
                                warning('Overriding fitrsvmOpts poly order = %d with svmRegressor poly order = %d',obj.order, varargin{i+1});
                            end
                            obj.order = varargin{i+1};
                            varargin{i+1} = [];
                        otherwise
                            warning('Option %s not supported', varargin{i});
                    end
                end
            end
            
            assert(isempty(obj.order) || strcmp(obj.kernel,'polynomial'), 'Must use polynomial kernel to specify polynomial order.');
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
            obj.Mdl = fitrsvm(double(X),Y, obj.fitrsvmOpts{:});
            
            if isa(obj.Mdl,'ClassificationPartitionedLinear')
                error('svmRegressor does not support using fitrsvm''s internal cross validation. Please wrap svmRegressor in a crossValScore() object instead.');
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function yfit_raw = score_samples(obj, X, varargin)
            yfit_raw = obj.Mdl.predict(double(X));
            if any(isnan(yfit_raw))
                warning('SVM model is returning nans. Check output, and don''t trust optimization algs');
            end
        end        
         
        function yfit_raw = score_null(obj, n)
            yfit_raw = repmat(obj.offset_null,n,1); 
        end
        
        function yfit = predict(obj, X)
            yfit = obj.Mdl.predict(double(X));
        end
        
        %% dependent methods
        
        function set.kernel(obj, val)          
            validKernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
            assert(ismember(val, validKernel), [sprintf('%s is an invalid kernel. ', val), ...
                                            'Valid kernels are:', sprintf(' ''%s''', validKernel{:})])
                                        
            kernel_idx = find(strcmp(obj.fitrsvmOpts, 'KernelFunction'));
            if isempty(kernel_idx)
                if ~strcmp(val,'linear') % no reason to populate extra options if we're using the default
                    obj.fitrsvmOpts = [obj.fitrsvmOpts, {'KernelFunction', val}];
                end
            else
                obj.fitrsvmOpts{kernel_idx + 1} = val;
            end
        end
        
        function val = get.kernel(obj)
            kernel_idx = find(strcmp(obj.fitrsvmOpts, 'KernelFunction'));
            if ~isempty(kernel_idx)            
                val = obj.fitrsvmOpts{kernel_idx+1};
            else
                val = 'linear';
            end
        end
        
        function set.epsilon(obj, val)
            if ~isempty(val)
                assert(val > 0, 'epsilon must be greater than 0');

                epsilon_idx = find(strcmp(obj.fitrsvmOpts,'Epsilon'));
                if isempty(epsilon_idx)
                    obj.fitrsvmOpts = [obj.fitrsvmOpts, {'Epsilon', val}];
                else
                    obj.fitrsvmOpts{epsilon_idx + 1} = val;
                end
            end
        end
        
        function val = get.epsilon(obj)
            epsilon_idx = find(strcmp(obj.fitrsvmOpts, 'Epsilon'));
            if ~any(epsilon_idx)
                val = [];
            else
                val = obj.fitrsvmOpts{epsilon_idx+1};
            end
        end
        
        function set.order(obj, val)          
            if ~isempty(val)
                assert(val > 0, 'Model order must be greater than zero');

                order_idx = find(strcmp(obj.fitrsvmOpts, 'PolynomialOrder'));
                if isempty(order_idx)
                    obj.fitrsvmOpts = [obj.fitrsvmOpts, {'PolynomialOrder', val}];
                else
                    obj.fitrsvmOpts{order_idx + 1} = val;
                end
            end
        end
        
        function val = get.order(obj)
            order_idx = find(strcmp(obj.fitrsvmOpts, 'PolynomialOrder'));
            assert(~isempty(order_idx), 'PolynomialOrder not specified in fitrsvmOpts. This is strange.');
            
            val = obj.fitrsvmOpts{order_idx+1};
        end
        
        function set.scale(obj, val)      
            if ~isempty(val)
                assert((ischar(val) && strcmp(val,'auto')) || val >= 0, 'kernel scale must be >=0 or ''auto''');

                scale_idx = find(strcmp(obj.fitrsvmOpts, 'KernelScale'));
                if isempty(scale_idx)
                    obj.fitrsvmOpts = [obj.fitrsvmOpts, {'KernelScale', val}];
                else
                    obj.fitrsvmOpts{scale_idx + 1} = val;
                end
            end
        end
        
        function val = get.scale(obj)
            scale_idx = find(strcmp(obj.fitrsvmOpts, 'KernelScale'));
            assert(~isempty(scale_idx), 'KernelScale not specified in fitrsvmOpts. This is strange.');
            
            val = obj.fitrsvmOpts{scale_idx+1};
        end
    end
end
