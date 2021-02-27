classdef lassoPcrRegressor < linearModelRegressor
    properties
        numcomponents = [];
        lambda = 0;
        alpha = 1;
        
        lassoParams = {};
    end
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'numcomponents', 'lambda', 'alpha'};
    end
    
    methods
        function obj = lassoPcrRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
                        case 'lambda'
                            obj.lambda = varargin{i+1};
                        case 'alpha'
                            obj.alpha = varargin{i+1};
                        case 'lassoParams'
                            obj.lassoParams = varargin{i+1};
                    end
                end
            end
            
            obj = obj.check_lasso_params();
        end
        
        function obj = fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj = obj.check_lasso_params();

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
                    disp('WARNING!! Number of components requested is more than unique components in training data.');
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

            if ~isempty(numcomps)
                [B, stats] = lasso(sc(:, 1:numcomps), Y, obj.lassoParams{:}, 'Alpha', obj.alpha, 'Lambda', obj.lambda);
            else
                [B, stats] = lasso(sc(:, 1:numcomps), Y, obj.lassoParams{:}, 'Alpha', obj.alpha, 'Lambda', obj.lambda);
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
            obj.offset = stats.Intercept;
            
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
    
    methods (Access = private)
        function obj = check_lasso_params(obj)
            % make adjustments to any lasso parameters in lasso options to
            % ensure compliance with expected lassoPcrRegressor usage.
            if any(strcmp(obj.lassoParams,'CV'))
                assert(isa(obj.lassoParams{find(strcmp(obj.lassoParams,'CV'))+1},'function_handel'), ...
                    ['CV object for lasso parameters must be a function handle of ',...
                    'type @(X,Y)(cvpartitioner(...)) that returns a CV partitioner ',...
                    'given input data (X,Y).']);
            end
            if any(strcmp(obj.lassoParams,'Alpha'))
                warning('Setting lassoCV Alpha based on alpha specified in lassoPcrRegressor constructor');
                alpha_idx = find(strcmp(obj.lassoParams,'Alpha'));
                obj.lassoParams(alpha_idx:alpha_idx+1) = [];
            end
            if any(strcmp(obj.lassoParams,'Lambda'))
                warning('Setting lassoCV Lambda based on lambda specified in lassoPcrRegressor constructor');
                lambda_idx = find(strcmp(obj.lassoParams,'Lambda'));
                obj.lassoParams(lambda_idx:lambda_idx+1) = [];
            end      
        end
    end
end

