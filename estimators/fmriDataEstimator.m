% obj = fmriDataEstimator(Estimator)
% In most cases fmriDataEstimators will serve as the primary vehical for
% Estimator implementation in this library. Exceptions are noted at the end
% of this help doc.
%   
%   properties (read only):
%       model       - modelEstimator or modelEstimator
%			representing underlying MVPA model, independent
%                       of any brain space parameterization
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
%       yfit = predict(obj, dat, [options])
%                   - Use modelEstimator to predict outcome in dat. 
%           options::
%               'fast' - true/false. If true fmriDataEstimator assumes that 
%                       dat.dat is already appropriately vectorized to be 
%                       passed through directly to modelEstimator. 
%                       Otherwise dat is transformed to match brainModel 
%                       before application of modelEstimator.
%               'featureConstructor_funhan'
%                       - Some models need metadata in addition to brain
%                       data (e.g. multilevelGlmRegressor needs some kind
%                       of identifier for random effects blocks like a
%                       subject indicator). features objects provide a way 
%                       of attaching this kind of metadata to matrix data,
%                       and this argument provides a way for the user to
%                       specify a function handle with instructions on how
%                       to construct this feature object. For instance if
%                       the necessary data is in
%                       fmri_data.metadata_table.subject_id then a sensible
%                       featureConstructor might be specified using an
%                       annonymous function like so,
%                       @(x1)(features(x1.dat', x1.metadata_table.subject_id)
%                       Note that x1.dat is transposed because model's
%                       expect input data to be obs x feature, while
%                       fmri_data objects are vxl x obs.
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
% We implement ML algoithms as modelEstimators which expect
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
        featureConstructor = @(X)(features(X.dat'));
                
        brainModel = fmri_data();
        
        isFitted = false;
        fitTime = -1;
    end

    properties (Access = ?Estimator)
        hyper_params = {};
    end
    
    methods   
        function obj = fmriDataEstimator(model, varargin)
            assert(isa(model,'modelEstimator'),...
                'Only modelEstimators are suppoted.')
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'featureConstructor_funhan'
                            obj.featureConstructor = varargin{i+1};
                    end
                end
            end
            
            obj.model = model;
        end
        
        function obj = fit(obj, dat, Y, varargin)
            if ~isa(obj.model, 'modelRegressor') || ~isa(obj.model, 'linearModelEstimator')
                warning(['Only (linearModelEstimator & modelRegressor) objects are fully supported models at this time.', ...
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
            X = obj.featureConstructor(dat);
            obj.model = obj.model.fit(X, Y, varargin{:});            
            
            obj.isFitted = true;
            
            obj.fitTime = toc(t0);
        end
        
        % similar to predict but calls obj.model.score_samples()
        function yfit_raw = score_samples(obj, dat, varargin)
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
                X = obj.featureConstructor(dat);
                yfit_raw = obj.model.score_samples(X);
            else
                if isa(obj.model, 'linearModelEstimator')
                    assert(isa(obj.model, 'modelRegressor'), ...
                        'Only modelRegressor linearModelEstimators are currently supported. You can try using predict() with the ''fast'' option but no guarantees of correct behavior.');
                    
                    weights = obj.get_weights();
                    yfit_raw = apply_mask(dat, weights, 'pattern_expression', 'dotproduct', 'none') + ...
                        obj.model.offset;
                else
                    error('Only linearModelEstimators are currently supported');
                end
            end
            yfit_raw = yfit_raw(:);
        end
                
        % similar to score_samples but calls obj.model.predict()
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
                X = obj.featureConstructor(dat);
                yfit = obj.model.predict(X);
            else
                if isa(obj.model, 'linearModelEstimator')
                    assert(isa(obj.model, 'modelRegressor'), ...
                        'Only modelRegressor linearModelEstimators are currently supported. You can try using predict() with the ''fast'' option but no guarantees of correct behavior.');
                    
                    weights = obj.get_weights();
                    yfit = apply_mask(dat, weights, 'pattern_expression', 'dotproduct', 'none') + ...
                        obj.model.offset;
                else
                    error('Only linearModelEstimators are currently supported');
                end
            end
            yfit = yfit(:);
        end
        
        
        function yfit_null = score_null(obj, varargin)                        
            yfit_null = obj.model.predict_null(varargin{:});
            
            yfit_null = yfit_null(:);
        end
        
        function yfit_null = predict_null(obj, varargin)                        
            yfit_null = obj.model.score_null(varargin{:});
            
            yfit_null = yfit_null(:);
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
