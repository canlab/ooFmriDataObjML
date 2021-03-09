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
%       weights = get_weights(obj)
%                   - combines brainModel with baseEstimator weights and
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
        featureConstructor = @(X)(features(X.dat'));
                
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
                        case 'featureConstructor_funhan'
                            obj.featureConstructor = varargin{i+1};
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
                weights = obj.get_weights();
                dat = obj.match_space(dat, weights);
            end
            
            dat = obj.featureConstructor(dat);
        end
        
        function weights = get_weights(obj)
            weights = obj.brainModel;
            
            weights.dat = ones(obj.datSize);
        end
    end
    
    methods (Static, Access = private)

        % this was largely copied from apply_mask in canlabCore
        % takes dat1 and ensures that it matches the space of dat2, then
        % returns dat2. dat2 should be weights as an fmri_data object,
        % using the fmri_data metdata from brainModel.
        function dat1 = match_space(dat1, dat2)
            isdiff = compare_space(dat1, dat2);

            if isdiff == 1 || isdiff == 2 % diff space, not just diff voxels
                % == 3 is ok, diff non-empty voxels

                % Both work, but resample_space does not require going back to original
                % images on disk
                dat2 = resample_space(dat2, dat1);

                % tor added may 1 - removed voxels was not legal otherwise
                %mask.removed_voxels = mask.removed_voxels(mask.volInfo.wh_inmask);
                % resample_space is not *always* returning legal sizes for removed
                % vox? maybe this was updated to be legal

                if length(dat2.removed_voxels) == dat2.volInfo.nvox
                    disp('Warning: resample_space returned illegal length for removed voxels. Fixing...');
                    dat2.removed_voxels = dat2.removed_voxels(dat2.volInfo.wh_inmask);
                end
            end
            

            % nonemptydat: Logical index of voxels with valid data, in in-mask space
            nonemptydat = get_nonempty_voxels(dat1);

            dat1 = replace_empty(dat1);

            % Check/remove NaNs. This could be done in-object...
            dat2.dat(isnan(dat2.dat)) = 0;

            % Replace if necessary
            dat2 = replace_empty(dat2);
            
            % save which are in mask, but do not replace with logical, because mask may
            % have weights we want to preserve
            inmaskdat = logical(dat2.dat);


            % Remove out-of-mask voxels
            % ---------------------------------------------------

            % mask.dat has full list of voxels
            % need to find vox in both mask and original data mask
%{
            if size(dat2.volInfo.image_indx, 1) == size(dat1.volInfo.image_indx, 1)
                n = size(dat2.volInfo.image_indx, 1);

                if size(nonemptydat, 1) ~= n % should be all vox OR non-empty vox
                    nonemptydat = zeroinsert(~dat1.volInfo.image_indx, nonemptydat);
                end

                if size(inmaskdat, 1) ~= n
                    inmaskdat = zeroinsert(~dat2.volInfo.image_indx, inmaskdat);
                end
                
                inboth = inmaskdat & nonemptydat;

                % List in space of in-mask voxels in dat object.
                % Remove these from the dat object
                to_remove = ~inboth(dat1.volInfo.wh_inmask);

            elseif size(dat2.dat, 1) == size(dat1.volInfo.image_indx, 1)

                % mask vox are same as total image vox
                nonemptydat = zeroinsert(~dat1.volInfo.image_indx, nonemptydat);
                inboth = inmaskdat & dat1.volInfo.image_indx & nonemptydat;

                % List in space of in-mask voxels in dat object.
                to_remove = ~inboth(dat1.volInfo.wh_inmask);

            elseif size(dat2.dat, 1) == size(dat1.volInfo.wh_inmask, 1)
                % mask vox are same as in-mask voxels in dat
                inboth = inmaskdat & dat1.volInfo.image_indx(dat1.volInfo.wh_inmask) & nonemptydat;

                % List in space of in-mask voxels in .dat field.
                to_remove = ~inboth;

            else
                fprintf('Sizes do not match!  Likely bug in resample_to_image_space.\n')
                fprintf('Vox in mask: %3.0f\n', size(dat2.dat, 1))
                fprintf('Vox in dat - image volume: %3.0f\n', size(dat1.volInfo.image_indx, 1));
                fprintf('Vox in dat - image in-mask area: %3.0f\n', size(dat1.volInfo.wh_inmask, 1));
            end

            dat1 = remove_empty(dat1, to_remove);
            %}
            
            % this ia dirty fix. What we need is to remove voxels that
            % aren't in the mask, but keep zero voxels from dat1 if they're
            % in the mask.
            if any(dat1.dat == 0)
                dat1 = dat1.cat(dat2);
                dat1 = remove_empty(dat1, ~inmaskdat);
                idx = size(dat1.dat,2);
                dat1 = dat1.get_wh_image(1:idx-1);
            else
                dat1 = remove_empty(dat1, ~inmaskdat);
            end
        end
    end
end


% shamelyessly ripped off from canlabCore's apply_mask()
function nonemptydat = get_nonempty_voxels(dat)
empty_voxels = all(dat.dat' == 0 | isnan(dat.dat'), 1)';

if size(empty_voxels, 1) == dat.volInfo.n_inmask
    % .dat is full in-mask length, we have not called remove_voxels or there are none to remove
    nonemptydat = ~empty_voxels;
    
elseif length(dat.removed_voxels) == dat.volInfo.n_inmask
    % .dat is not in-mask length, and we have .removed_voxels defined
    nonemptydat = false(dat.volInfo.n_inmask, 1);
    nonemptydat(~dat.removed_voxels) = true;
    
    % additional: we could have invalid voxels that have been changed/added since
    % remove_empty was last called.
    %     dat.removed_voxels(dat.removed_voxels) =
    %     nonemptydat(empty_voxels
end

end
