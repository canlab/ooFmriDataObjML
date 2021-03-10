% lassoPcrRegressor A linearModelEstimator that implements lasso and
%   elastic-net PCR
% 
% m = lassoPcrRegressor(options)
%
%   options::
%       'alpha' - Default: 1.
%
%       'lambda' - Default: 0. Specify [] to use lasso() defaults;
%
%       'numcomponents' - Default: Max. Number of PCR components to retain.
%
%       'lassoParams' - arguments to passthrough to matlab's lasso()
%
% Both lasso and lassoPcrRegressor have hyperparameter specification
% methods. This function changes the default behavior of lasso with respect
% to these parameters. If you specify different parameters in the
% lassoPcrRegressor constructor and in the lassoParams arguments, the
% lassoPcrRegressor parameters will take precedence. To compute the LARS
% path and pick the MSE minimizing value explicitly specifity 'lasso', [].
% By default no cross validation is performed. You should specify a
% cvpartition object so that the CV estimate of MSE is minimized instead.
% Alternatively you can supply a cvpartition generator (a function handle
% that returns a cvpartition object) in lassoParams, and lassoPcrRegressor
% will generate cvpartition objects on demand.
%
% Examples ::
% Note that for simplicity the data is passed in vectorized form here using
% features objects, but lassoPcr could also be wrapped in an
% fmriDataEstimator object with an appropriately specified
% featuresConstructor if you wanted to develop an baseEstimator that operated
% directly on fmri_data objects.
% 
%
% dat; % an fmri_data object with subject blocks identified by a
%      % dat.metadata_table.subject_id field, and outcome data in dat.Y
%
%   Example 1:
%   % examle using all components and the LARS algorithm with cross validation
%   % for Lambda selection
% 
%     dat_feat = features(dat.dat', dat.metadata_table.subject_id);
%     cvpartitioner = @(X,Y)cvpartition2(ones(size(Y)), 'KFOLD', 5, 'Stratify', X.metadata);
% 
%     lassoPcr = lassoPcrRegressor('lambda', [], 'lassoParams', {'CV', cvpartitioner});
% 
%     % get self-fit
%     lassoPcr = lassoPcr.fit(dat_feat, dat.Y);
%     yfit = lassoPcr.predict(dat_feat);
%     plot(dat.Y, yfit, '.')
% 
%     % get cross validated fit
%     cvPredictor = crossValScore(lassoPcr, cvpartitioner, @get_mse, ...
%        'repartOnFit', false, 'n_parallel', 1, 'verbose', false);
% 
%     cvPredictor = cvPredictor.do(dat_feat, dat.Y);
%     cvPredictor = cvPredictor.do_null;
%     f = figure(2);
%     cvPredictor.plot('Parent',gca(f));
% 
% 
%   Example 2: 
%     % example using all components and bayesian optimization for Lambda
%     % selection
% 
%     dat_feat = features(dat.dat', dat.metadata_table.subject_id);
%     cvpartitioner = @(X,Y)cvpartition2(ones(size(Y)), 'KFOLD', 5, 'Stratify', X.metadata);
% 
%     lassoPcr = lassoPcrRegressor();
% 
%     lambda = optimizableVariable('lambda',[0.001,1000],'Type','real','Transform','log');
%     bayesOptOpts = {lambda, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
%         'MaxObjectiveEvaluations', 10, 'UseParallel', 1, 'verbose', 0, 'PlotFcn', {}};
%     bo_lassopcr = bayesOptCV(lassoPcr, cvpartitioner, @get_mse, bayesOptOpts);
% 
%     % self fit
%     bo_lassopcr = bo_lassopcr.fit(dat_feat, dat.Y);
%     yfit = bo_lassopcr.predict(dat_feat);
%     plot(dat.Y, yfit, '.')
% 
%     % get cross validated fit
%     cvPredictor = crossValScore(bo_lassopcr, cvpartitioner, @get_mse, ...
%        'repartOnFit', false, 'n_parallel', 1, 'verbose', false);
% 
%     cvPredictor = cvPredictor.do(dat_feat, dat.Y);
%     cvPredictor = cvPredictor.do_null;
%     f = figure(2);
%     cvPredictor.plot('Parent',gca(f));
% 
% 
%   Example 3:
%     % example using bayesian optimization to select number of components in an
%     % outer CV loop and LARS to select Lambda in an inner CV loop.
% 
%     dat_feat = features(dat.dat', dat.metadata_table.subject_id);
%     cvpartitioner = @(X,Y)cvpartition2(ones(size(Y)), 'KFOLD', 5, 'Stratify', X.metadata);
% 
%     lassoPcr = lassoPcrRegressor('lambda', [], 'lassoParams', {'CV', cvpartitioner});
% 
%     dims = optimizableVariable('numcomponents',[1,77],'Type','integer');
%     bayesOptOpts = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
%         'MaxObjectiveEvaluations', 10, 'UseParallel', 1, 'verbose', 0, 'PlotFcn', {}};
%     bo_lassopcr = bayesOptCV(lassoPcr, cvpartitioner, @get_mse, bayesOptOpts);
% 
%     % self fit
%     bo_lassopcr = bo_lassopcr.fit(dat_feat, dat.Y);
%     yfit = bo_lassopcr.predict(dat_feat);
%     plot(dat.Y, yfit, '.')
% 
%     % get cross validated fit
%     cvPredictor = crossValScore(bo_lassopcr, cvpartitioner, @get_mse, ...
%        'repartOnFit', false, 'n_parallel', 1, 'verbose', false);
% 
%     cvPredictor = cvPredictor.do(dat_feat, dat.Y);
%     cvPredictor = cvPredictor.do_null;
%     f = figure(2);
%     cvPredictor.plot('Parent',gca(f));
classdef lassoPcrRegressor < linearModelEstimator & modelRegressor    
    properties (Dependent, SetAccess = ?baseEstimator)
        lambda
        alpha
    end
    
    properties (SetAccess = protected)                
        isFitted = false;
        fitTime = -1;  
        lassoParams = {};

        B = [];
        offset = 0;
        offset_null = 0;
         
        lassoCV_funhan = [];
    end
    
    properties(SetAccess = ?baseEstimator)
        numcomponents = [];
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {'numcomponents', 'lambda', 'alpha'};
    end
    
    methods
        function obj = lassoPcrRegressor(varargin)
            % set this first, since other parameters will override values
            % here if there's a conflict, and we don't want these values
            % intead overriding other lassoPcrRegressor values
            lassoParams_idx = find(strcmp(varargin,'lassoParams'));
            if ~isempty(lassoParams_idx)
                obj.lassoParams = varargin{lassoParams_idx + 1};
            end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
                        case 'lambda'
                            obj.lambda = varargin{i+1};
                        case 'alpha'
                            obj.alpha = varargin{i+1};
                    end
                end
            end
            
            % if hyperparameters were not set at any point, default to
            % these. (To get default lasso() behavior you need to
            % explicitly specify empty vectors for hyperparameters
            % somewhere)
            if isempty(obj.alpha)
                obj.alpha = 1;
            end
            if ~any(strcmp(obj.lassoParams,'Lambda'))
                obj.lambda = 0;
            end
            
            obj.check_lasso_params();
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);

            obj.check_lasso_params();

            % code below was copied from fmri_data/predict with minor
            % modifications for our different numenclature.
            [pc,~,~] = svd(scale(X,1)', 'econ'); % replace princomp with SVD on transpose to reduce running time. 
            pc(:,end) = [];                % remove the last component, which is close to zero.
                                           % edit:replaced 'pc(:,size(xtrain,1)) = [];' with
                                           % end to accomodate predictor matrices with
                                           % fewer features (voxels) than trials. SG
                                           % 2017/2/6                              
                               
            % Choose number of components to save [optional]
            if ~isempty(obj.numcomponents)

                numc = obj.numcomponents;

                if obj.numcomponents > size(pc, 2)
                    warning('Number of components requested (%d) is more than unique components in training data (%d)', obj.numcomponents, size(pc,2));
                    numc = size(pc, 2);
                end
                pc = pc(:, 1:numc);
            end

            sc = X * pc;

            if rank(sc) == size(sc,2)
                numcomps = rank(sc); 
            elseif rank(sc) < size(sc,2)
                numcomps = rank(sc)-1;
            end

            % if there's a cvpartition generator available, invoke it
            lassoParams = obj.lassoParams;
            if ~isempty(obj.lassoCV_funhan)
                cvpart = obj.lassoCV_funhan(X,Y);
                cv_idx = find(strcmp(lassoParams,'CV'));
                
                % do a sanity check. We get the function handle from the
                % 'CV' argument whenever check_lasso_params() is called, so
                % if we have one but not the other something very strange
                % is happening.
                assert(~isempty(cv_idx), 'lassoCV_function_handle was found, but no ''CV'' parameter was found in lassoParams. Something is wrong.');
                lassoParams{cv_idx+1} = cvpart;
            end
            
            if ~isempty(numcomps)
                [B, stats] = lasso(sc(:, 1:numcomps), Y, lassoParams{:});
            else
                [B, stats] = lasso(sc, Y, lassoParams{:});
            end
            
            if size(B,2) > 1
                lambda_idx = find(stats.MSE == min(stats.MSE));
                if ~any(strcmp(obj.lassoParams,'CV'))
                    warning('No lambda value or cvpartitioner provided. Using MSE minimizing Lambda on nonCV training data: Lambda = %0.3f.', ...
                        stats.Lambda(lambda_idx));
                else
                    fprintf('Using MSE minimizing Lambda based on CV LARS perf est, Lambda = %0.3f\n',...
                        stats.Lambda(lambda_idx));
                end
                assert(~isempty(lambda_idx),'No optimal Lambda could be found! Something strange is happening.');
                B = B(:,lambda_idx);
            end
                
            wh_beta = logical(B~=0);
            
            Xsubset = [ones(size(sc, 1), 1) sc(:, wh_beta)];
            betatmp = pinv(Xsubset) * Y;
            betas = zeros(size(wh_beta));
            betas(wh_beta) = betatmp(2:end);

            
            if ~isempty(numcomps)
                obj.B = pc(:, 1:numcomps) * betas;
            else
                obj.B = pc* betas;
            end
            obj.offset = betatmp(1);
            
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        
        % we need to modify this logic a bit to keep lassoParams in sync 
        % with lassoPcrRegressor hyperparameters 
        function set_params(obj, hyp_name, hyp_val)
            set_params@baseEstimator(obj, hyp_name, hyp_val);
            
            % this will adjust lassoParams to match the hyperparameters
            % specified.
            % turn off warnings because at this stage we expect
            % discrepancy, that's why we're calling this function, to fix 
            % it.
            warning('off','lassoPcrRegressor:param_check');
            check_lasso_params(obj);
            warning('on','lassoPcrRegressor:param_check');
        end       
        
        
        
        
        function set.alpha(obj, val)
            alpha_idx = find(strcmp(obj.lassoParams,'Alpha'));
            if isempty(alpha_idx)
                obj.lassoParams = [obj.lassoParams, {'Alpha', val}];
            else
                obj.lassoParams{alpha_idx+1} = val;
            end
        end
        
        function val = get.alpha(obj)
            alpha_idx = find(strcmp(obj.lassoParams,'Alpha'));
            if isempty(alpha_idx)
                val = [];
            else
                val = obj.lassoParams{alpha_idx+1};
            end
        end
        
        function set.lambda(obj, val)
            lambda_idx = find(strcmp(obj.lassoParams,'Lambda'));
            if isempty(lambda_idx)
                obj.lassoParams = [obj.lassoParams, {'Lambda', val}];
            else
                obj.lassoParams{lambda_idx+1} = val;
            end
        end
        
        function val = get.lambda(obj)
            lambda_idx = find(strcmp(obj.lassoParams,'Lambda'));
            if isempty(lambda_idx)
                val = [];
            else
                val = obj.lassoParams{lambda_idx+1};
            end
        end
    end
    
    methods (Access = private)
        % LassoCV has an argument for specifying cross
        % validation folds, and this check incorporates a method for
        % allowing the user to specify function handles to cvpartition 
        % object generators instead of cvpartition instances. This is
        % useful if this baseEstimator ends up wrapped in some
        % crossValidator object, since passing a function handle then 
        % allows for cvpartition to be generated on demand based on the
        % particular fold slicing that's received from the crossValidator.
        function check_lasso_params(obj)     
            cv_idx = find(strcmp(obj.lassoParams,'CV'));
            if ~isempty(cv_idx)
                if isa(obj.lassoParams{cv_idx+1},'function_handle')
                    obj.lassoCV_funhan = obj.lassoParams{cv_idx+1};
                end
            end
        end
    end
end

