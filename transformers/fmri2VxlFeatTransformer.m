% obj = fmri2VxlFeatTransformer(Estimator)
% In most cases fmri2VxlFeatTransformers will serve as the primary vehical for
% Estimator implementation in this library. Exceptions are noted at the end
% of this help doc.
%   
%   properties (read only):
%       model       - baseEstimator representing underlying MVPA model, independent
%                       of any brain space parameterization
%       brainModel  - an fmri_data object that represents the model's 
%                       corresponding brain space.
%
%   methods:
%       obj = fit(obj, dat, Y, varargin)
%                   - fit a model to voxel data of fmri_data object dat to
%                       predict Y using the baseEstimator specified when 
%                       the constructor was invoked. Any additional 
%                       arguments are passed through to the baseEstimator.
%
%       yfit = predict(obj, dat, [options])
%                   - Use baseEstimator to predict outcome in dat. 
%           options::
%               'fast' - true/false. If true fmri2VxlFeatTransformer assumes that 
%                       dat.dat is already appropriately vectorized to be 
%                       passed through directly to baseEstimator. 
%                       Otherwise dat is transformed to match brainModel 
%                       before application of baseEstimator.
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
%       weightMask = get_weightMask(obj)
%                   - combines brainModel with baseEstimator weightMask and
%                      returns the model in brain space. Useful for pattern
%                      visualization.
%
%       params = get_params(obj)
%                   - returns a list of valid hyperparameters for
%                      baseEstimator.
%
%       obj = set_params(obj, hyp_name, hyp_val)
%                   - sets hyperparameter value of hyp_name to hyp_val for
%                       baseEstimator.
% 
% We implement ML algoithms as baseEstimators which expect
% vectorized input, but fmri_data is not vectorized in a predictable way. 
% fmri_data objects can differ in resolution and spatial extent for 
% instance. fmri2VxlFeatTransformers serve as wrappers that mediate between 
% fmri_data input and vectorized input.
%
% Separating the ML algorithm from the fmri_data handling is useful because
% it allows for more flexible feature construction (through transformers
% and pipelines). The final feature set does not need to be an fmri_data
% object, and in that case you can simply invoke linearModelEstimators
% directly rather than through an fmri2VxlFeatTransformer.

classdef fmri2VxlFeatTransformer < baseTransformer
    properties (SetAccess = private)
        metadataConstructor = @(X)(features(X.dat'));
                
        brainModel = fmri_data();
        datSize = [1,1];
        
        isFitted = false;
        fitTime = -1;
    end
    
    properties (Access = ?baseTransformer)
        hyper_params = {};
    end
    
    methods   
        function obj = fmri2VxlFeatTransformer(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'metadataConstructor_funhan'
                            obj.metadataConstructor = varargin{i+1};
                        otherwise
                            warning('Did not understand input %s to fmri2VxlFeatTransformer', varargin{i});
                    end
                end
            end
        end
        
        function fit(obj, dat, varargin)            
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
            obj.datSize = size(obj.brainModel.dat);
            obj.brainModel.dat = [];            
            
            obj.brainModel.history = {[class(obj), ' fit']};
            
            obj.isFitted = true;
            
            obj.fitTime = toc(t0);
        end
        
        % similar to predict but calls obj.model.score_samples()
        function dat = transform(obj, dat, varargin)
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
            assert(obj.isFitted, sprintf('Please call %s.fit() before %s.transform().\n',class(obj)));
            if ~fast
                weightMask = obj.get_weightMask();
                
                dat = obj.match_space(dat, weightMask);
            end
            
            dat = features(dat.dat', obj.metadataConstructor(dat));
        end
        
        function weightMask = get_weightMask(obj)
            weightMask = obj.brainModel;
            
            weightMask.dat = ones(obj.datSize);
        end
    end
    
    methods (Static)
        function dat = match_space(dat, mask)
            
            isdiff = compare_space(dat, mask);

            if isdiff == 1 || isdiff == 2 % diff space, not just diff voxels
                % == 3 is ok, diff non-empty voxels

                dat = resample_space(dat, mask);

                if length(dat.removed_voxels) == dat.volInfo.nvox
                    warning('resample_space returned illegal length for removed voxels. Fixing...');
                    dat.removed_voxels = dat.removed_voxels(dat.volInfo.wh_inmask);
                end
            end

            dat = replace_empty(dat);

            % Replace if necessary
            mask = replace_empty(mask);

            inmaskdat = logical(mask.dat);

            
            % Remove out-of-mask voxels
            % ---------------------------------------------------

            % mask.dat has full list of voxels
            % need to find vox in both mask and original data mask

            if size(mask.volInfo.image_indx, 1) == size(dat.volInfo.image_indx, 1)
                n = size(mask.volInfo.image_indx, 1);

                if size(inmaskdat, 1) ~= n
                    inmaskdat = zeroinsert(~mask.volInfo.image_indx, inmaskdat);
                end

                % List in space of in-mask voxels in dat object.
                % Remove these from the dat object
                to_remove = ~inmaskdat(dat.volInfo.image_indx);

            elseif size(mask.dat, 1) == size(dat.volInfo.image_indx, 1)
                
                to_remove = ~inmaskdat(dat.volInfo.wh_inmask);

            elseif size(mask.dat, 1) == size(dat.volInfo.wh_inmask, 1)
              
                to_remove = ~inmaskdat;

            else
                fprintf('Sizes do not match!  Likely bug in resample_to_image_space.\n')
                fprintf('Vox in mask: %3.0f\n', size(mask.dat, 1))
                fprintf('Vox in dat - image volume: %3.0f\n', size(dat.volInfo.image_indx, 1));
                fprintf('Vox in dat - image in-mask area: %3.0f\n', size(dat.volInfo.wh_inmask, 1));
            end

            % keep overall list
            dat.removed_voxels = to_remove;
            dat.dat(to_remove,:) = [];
        end
    end
end