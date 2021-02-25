% obj = fmriDataEstimator(modelEstimator)
% In most cases fmriDataEstimators will serve as the primary vehical for
% Estimator implementation in this library. Exceptions are noted at the end
% of this help doc.
%   
%   properties (read only):
%       model       - modelEstimator representing underlying MVPA model,
%                       independent of any brain space parameterization
%       brainModel  - an fmri_data object that represents the model's 
%                       corresponding brain space.
%
%   methods:
%       obj = fit(obj, dat, Y, varargin)
%                   - fit a model to voxel data of fmri_data object dat to
%                       predict Y using the modelEstimator specified when 
%                       the constructor was invoked. Any additional 
%                       arguments are passed through to the modelEstimator.
%
%       yfit = predict(obj, dat, ['fast', true/false])
%                   - Use modelEstimator to predict outcome in dat. If
%                       'fast', is set to true the fmriDataEstimator
%                       assumes that dat.dat is already appropriately
%                       vectorized to be passed through directly to
%                       modelEstimator. Otherwise dat is transformed to
%                       match brainModel before application of
%                       modelEstimator.
%
%       weights = get_weights(obj)
%                   - combines brainModel with modelEstimator weights and
%                      returns the model in brain space. Useful for pattern
%                      visualization.
%
%       params = get_params(obj)
%                   - returns a list of valid hyperparameters for
%                      modelEstimator.
%
%       obj = set_hyp(obj, hyp_name, hyp_val)
%                   - sets hyperparameter value of hyp_name to hyp_val for
%                       modelEstimator.
% 
% We implement ML algoithms as linearModelEstimators which expect
% vectorized input, but fmri_data is not vectorized in a predictable way. 
% fmri_data objects can differ in resolution and spatial extent for 
% instance. fmriDataEstimators serve as wrappers that mediate between 
% fmri_data input and vectorized input.
%
% Separating the ML algorithm from the fmri_data handling is useful because
% it allows for more flexible feature construction (through transformers
% and pipelines). The final feature set does not need to be an fmri_data
% object, and in that case you can simply invoke linearModelEstimators
% directly rather than through an fmriDataEstimator.

classdef fmriDataEstimator < Estimator
    properties (SetAccess = private)
        model = @plsRegressor
                
        brainModel = fmri_data();
        
        isFitted = false;
        fitTime = -1;
    end

    properties (Access = ?Estimator)
        hyper_params = {};
    end
    
    methods   
        function obj = fmriDataEstimator(model)
            assert(isa(model,'modelEstimator'), 'Only modelEstimators are suppoted.')
            
            obj.model = model;
        end
        
        function obj = fit(obj, dat, Y, varargin)
            if ~isa(obj.model, 'linearModelRegressor')
                warning(['Only linearModelRegressor objects are fully supported models at this time.', ...
                    'You will need to call %s.predict() with the ''fast'' option to obtain ',...
                    'predictions with unsupported model objects.'], obj(class));
            end
            
            t0 = tic;
            % get prototype image
            % this needs a solution for when image 1 differs in voxel count
            % from the full dataset
            obj.brainModel = fmri_data(dat.get_wh_image(1));
            fnames = {'images_per_session', 'Y', 'Y_names', 'Y_descrip', 'covariates',...
                'additional_info', 'metadata_table', 'dat_descrip', 'image_names', 'fullpath'};
            for field = fnames
                obj.brainModel.(field{1}) = [];
            end
            obj.brainModel.dat = [];
            
            obj.brainModel.history = {[class(obj), ' fit']};
            
            % this is where the MVPA model is actually fit
            obj.model = obj.model.fit(dat.dat', Y, varargin{:});            
            
            obj.isFitted = true;
            
            obj.fitTime = toc(t0);
        end
        
        function yfit = predict(obj, dat, varargin)
            fast = false;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'fast'
                            fast = varargin{i+1};
                    end
                end
            end
                            
            % we have a fast option and a slow option.
            % fast option - will work if dat is in the same space as
            %   obj.brainModel(). Good for faster cross validation.
            % slow option - will transform dat to be in obj.brainModel()'s
            %   space, so more robust, but slower. Good for testing models
            %   on novel data.
            assert(obj.isFitted, sprintf('Please call %s.fit() before %s.predict().\n',class(obj)));
            if fast
                yfit = obj.model.predict(dat.dat);
            else
                if isa(obj.model, 'linearModelEstimator')
                    assert(isa(obj.model, 'linearModelRegressor'), ...
                        'Only linearModelRegressors are currently supported. You can try using predict() with the ''fast'' option but no guarantees of correct behavior.');
                    
                    weights = obj.get_weights();
                    yfit = apply_mask(dat, weights, 'pattern_expression', 'dotproduct', 'none') + ...
                        obj.model.offset;
                else
                    error('Only linearModelEstimators are currently supported');
                end
            end
            yfit = yfit(:);
        end
        
        % fmriDataEstimator's would take no hyperparameters, and simply
        % pass through whatever the underlying functions hyperparameters 
        % are.
        function params = get_params(obj)
            params = obj.model.get_params();
        end
        
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s must be a hyperparameter of %s\n', hyp_name, class(obj.model)));
            
            obj.model = obj.model.set_hyp(hyp_name, hyp_val);
        end
        
        function weights = get_weights(obj)
            weights = obj.brainModel;
            weights.dat = obj.model.B(:);
        end
    end
end
