% implements reference batch ComBat harmonization. ComBat parameter estimation
% is sensitive to certain types of data charactreistics. sani_for_combat is used
% to modify the data before combat is used. Please refer to that function in
% utils for details
classdef fmriDataCombatTransformer < baseTransformer
    properties
        combat_opts = {[], 1};
    end
    
    properties (SetAccess = private)
        get_batch_id = @(X)(X.metadata_table.subject_id);
        
        % reference batch info
        ref_params;
        ref_batch_id;   
        
        isFitted = false;
        fitTime = -1;
    end
    
    properties (Access = ?baseTransformer)
        hyper_params = {'parametric'};
    end
    
    methods
        function obj = fmriDataCombatTransformer(batch_id_funhan, combat_opts)
            if any(cellfun(@(x1)(ischar(x1) && strcmp(x1,'ref_idx')), combat_opts))
                idx = find(cellfun(@(x1)(ischar(x1) && strcmp(x1,'ref_idx')), combat_opts));
                combat_opts{idx:idx+1} = [];
                warning('Found ref_idx in combat opts, removing it...\n');
            end
            obj.combat_opts = combat_opts;
            
            obj.get_batch_id = batch_id_funhan;
        end
        
        function fit(obj, dat, varargin)
            t0 = tic;
            batch_id = categorical(obj.get_batch_id(dat));
            
            % make sure dat and batch id have sensible sizes
            assert(size(dat.dat,2) == length(batch_id), 'dat be length(bach_id) x m');
            assert(ismatrix(dat.dat));
                      
            min_n = 3;
            for i = 1:length(obj.combat_opts)
                if ischar(obj.combat_opts{i}) && strcmp(obj.combat_opts{i},'meanOnly')
                    if varargin{i+1} == 1
                        min_n = 2;
                    end
                end
            end
            [uniq_batch,b] = unique(batch_id);
            b(end+1) = length(batch_id)+1;
            n_batches = diff(b);
            bad_idx = find(n_batches < min_n);
            if any(bad_idx)
                warning('We should have minimum %d df to apply combat, but batch %s (and possibly others) only has %d instances.', ...
                    min_n, char(uniq_batch(bad_idx(1))), n_batches(bad_idx(1)));
            end
            
            obj.pick_ref_batch(dat, batch_id);            
            [~, grand_mean, var_pooled, B_hat] = evalc(['obj.get_ref_batch_params(',...
                'dat.dat, batch_id, obj.combat_opts{:}, ''ref'', obj.ref_batch_id)']);
                        
            % save combat parameters as fmri_data objects for easy
            % resampling, masking anywhere reference image has zero
            % variance, because we can't combat correct these.
            obj.ref_params = dat.get_wh_image(1:size(B_hat,1)+2);
            obj.ref_params.dat = [grand_mean(:), var_pooled, B_hat'];
            
            varMask = fmri_mask_image(obj.ref_params.get_wh_image(2));
            varMask.dat(varMask.dat == 0) = 0;
            obj.ref_params = obj.ref_params.apply_mask(varMask);
            
            fnames = {'images_per_session', 'Y', 'Y_names', 'covariates',...
                'additional_info', 'metadata_table', 'dat_descrip', 'image_names', 'fullpath'};
            for field = fnames
                obj.ref_params.(field{1}) = [];
            end
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        % Note: This function will behave differently when transforming a
        % single batch (e.g. in a prediction application) vs. when
        % transforming multiple batches. The reason being that it aims for
        % a homogenous brain mask across all batch instances in dat. Some
        % batches will have invalid voxels in some areas, which will
        % consequently be removed from all batches for instance. There is
        % room for further development here to fix this issue.
        function cb_dat = transform(obj, dat)
            assert(obj.isFitted,'Please call combatTransformer.fit() before combatTransformer.transform().');
            
            batch_id = categorical(obj.get_batch_id(dat));
            
            % make sure dat and batch id have sensible sizes
            assert(size(dat.dat,2) == length(batch_id), 'dat be length(bach_id) x m');
            if isvector(dat.dat)
                idx = cellfun(@(x1)(ischar(x1) && strcmp(x1,'meanOnly')), obj.combat_opts);
                assert(~isempty(idx) && obj.combat_opts{idx+1} == 1, ...
                    'dat is a vector, but meanOnly not selected for combat.')
            else
                assert(ismatrix(dat.dat));
            end
            
            % check that we have necessary degrees of freedom
            %{
            %}
            
            % append reference batch params to list to remove problematic 
            % voxels in side by side comparison
            n = size(dat.dat,2);
            dat = dat.cat(obj.ref_params).remove_empty();
            
            batch_id = categorical(batch_id);
            tmp_ref_id = categorical(max(double(batch_id)) + 1);
            while ismember(tmp_ref_id, batch_id)
                % make sure this is unique, should be but just in case
                tmp_ref_id = categorical(double(tmp_ref_id) + 1);
            end
            batch_id = [batch_id(:)', repmat(tmp_ref_id,1,size(obj.ref_params.dat,2))];
            
            % 'sanitizes' data for combat
            dat = sani_for_combat(dat, batch_id);
            dat = dat.remove_empty();
            
            % retrieve the spatially updated params
            grand_mean = dat.dat(:,n+1)';
            var_pooled = dat.dat(:,n+2);
            if size(dat.dat,2) > n + 2
                B_hat = dat.dat(:,n+3:end)';
            else
                B_hat = [];
            end
            
            % apply combat to subset of data that doesn't include the
            % appended params (supply params explicitly as fxn args)
            [~, dat.dat(:,1:n)] = evalc(['obj.refCombat(grand_mean, var_pooled, B_hat, ',...
                    'dat.dat(:,1:n), batch_id(1:n), obj.combat_opts{:})']);
            assert(all(all(~isnan(dat.dat))),...
                'One of the images returned all NaNs after combat correction. This may indicate a bug in sani_for_combat(), or some gross irregularity in your data, some kind of missing data perhaps?.');
            
            % remove reference batch
            cb_dat = dat.get_wh_image(batch_id ~= tmp_ref_id);
        end
        
        function set_params(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            if ismember(hyp_name, 'parametric')
                assert(ismember(hyp_val,[0,1]), 'parametric flag must be 0/1')
                obj.combat_opts{2} = hyp_val;
            end
        end
    end
        
    methods (Access = private)
        function pick_ref_batch(obj, dat, batch_id)
            [uniq_batch_id, exp_batch_id] = unique(batch_id,'stable');

            % skip picking representative batch if we only have one batch
            if length(exp_batch_id) == 1
                obj.ref_batch_id = exp_batch_id(1);
                return 
            end
            
            % now let's pick a representative batch to use as the reference
            % given a fixed training set this selection is deterministic
            cmat = []; % within batch centering matrix
            mumat = []; % within batch averaging matrix;
            n = [];
            for j = 1:length(uniq_batch_id)
                this_blk = uniq_batch_id(j);
                this_n = sum(this_blk == batch_id);
                n = [n(:); this_n*ones(this_n,1)];
                cmat = blkdiag(cmat, eye(this_n) - 1/this_n);
                mumat = blkdiag(mumat, ones(this_n)*1/this_n);
            end

            % initialize fmri_dat obj's for storage of mdat and sdat info
            [mdat, sdat] = deal(dat.get_wh_image(1:length(uniq_batch_id)));

            sdat.dat = ((dat.dat*cmat).^2)*mumat;
            sdat.dat = sdat.dat(:,exp_batch_id).^0.5; % this is the voxel-wise std for each subject

            mdat.dat = dat.dat*mumat;
            mdat.dat = mdat.dat(:,exp_batch_id); % this is voxel-wise mean for each subject

            % sdat.^2 and mdat span the set of reference values combat can
            % normalize to, so let's make sure we pick a typical batch, not some
            % outlier.

            % compute pattern regularity according to mahalanobis distance
            [~,~,~,mp] = evalc('mahal(mdat,''corr'',''noplot'')'); 
            [~,I] = sort(mp,'descend');
            [~,mI] = sort(I); % mI tells you the rank of mp, with smaller numbers having bigger p values (more typical)
            % compute variance similarity according to mahalanobis distance
            [~,~,~,sp] = evalc('mahal(sdat,''noplot'')');
            [~,I] = sort(sp,'descend');
            [~,sI] = sort(I); % mI tells you the rank of mp, with smaller numbers having bigger p values (more typical)
    
            typ = sI + mI; % weigh variance and mean typicality evenly
            obj.ref_batch_id = uniq_batch_id(find(typ == min(typ),1,'first'));
        end
        
        function [grand_mean, var_pooled, B_hat] = get_ref_batch_params(~, dat, batch, mod, ~, varargin)
            if any(isnan(dat(:)))
                error('Input data contains nan entries. These will break combat. Please fix and rerun');
            end

            dat = double(dat); % there are EB algorithm convergence issues if input data is single
            [sds] = std(dat')';
            wh = find(sds==0);
            [ns,~] = size(wh);
            if ns>0
                error('Error. There are rows with constant values across samples. Remove these rows and rerun.')
            end
            if ischar(batch) % needed for sort function to work reliably
                batch = cellstr(batch);
            end

            ref = [];
            meanOnly = false;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'ref'                
                            ref = varargin{i+1};
                            if ~ismember(ref,batch)
                                error('Reference is not in batch list, please check inputs');
                            else
                                if ischar(ref) || iscategorical(ref)
                                    fprintf('[combat] Harmonizing to reference batch %s\n', ref);
                                elseif isnumeric(ref) || islogical(ref)
                                    fprintf('[combat] Harmonizing to reference batch %d\n', ref);
                                else
                                    error('Ref argument must be type numeric or char\n');
                                end
                            end
                        case 'meanOnly'
                            if varargin{i+1} == 1
                                meanOnly = true;
                            elseif varargin{i+1} == 0
                                meanOnly = false;
                            else
                                error('Did not understand argument to meanOnly');
                            end

                    end
                end
            end
            
            assert(~isempty(ref),'Only reference combat harmonization is supported.');

            batchmod = categorical(batch);
            batchmod = dummyvar({batchmod});
            n_batch = size(batchmod,2);
            uniq_batch = unique(batch,'stable');

            batches = cell(0);
            for i=1:n_batch
                batches{i}=find(batch == uniq_batch(i));
            end
            n_batches = cellfun(@length,batches);
            n_array = sum(n_batches);

            % Creating design matrix and removing intercept:
            if ~isempty(mod)
                % you should check that the following code evaluates to
                % true
                %
                % ct = combatTransformer2()
                % new_ct_dat = ct.fit_transform(dat); 
                % orig_ct_dat = combat(fmri_data.dat, batch_id, design, parametric, 'ref', ct.ref_batch_id)
                % all(diag(corr(new_ct_dat, orig_ct_dat)))
                %
                % modified so that combatTransformer2 takes design as input
                
                warning('ComBat with regressors has not been tested. Please compare your results with Fortin''s ComBat implementations');
            end
            design = [batchmod mod];
            intercept = ones(1,n_array)';
            wh = cellfun(@(x) isequal(x,intercept),num2cell(design,1));
            bad = find(wh==1);
            design(:,bad)=[];

            if ~isempty(ref)
                design(:,ismember(uniq_batch,ref)) = 1;
            end

            fprintf('[combat] Computing adjustments for %d covariate(s) of covariate level(s)\n',size(design,2)-size(batchmod,2))
            % Check if the design is confounded
            if rank(design)<size(design,2)
                nn = size(design,2);
                if nn==(n_batch+1) 
                  error('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
                end
                if nn>(n_batch+1)
                  temp = design(:,(n_batch+1):nn);
                  if rank(temp) < size(temp,2)
                    error('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
                  else 
                    error('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')
                  end
                end
             end

            B_hat = inv(design'*design)*design'*dat';
            
            %Standarization Model
            grand_mean = B_hat(ismember(uniq_batch,ref),:);

            ref_dat = dat(:,ismember(batch,ref));
            ref_n = n_batches(ismember(uniq_batch,ref));
            % this is useful for diagnostics, but don't leave it because parallel workers 
            % will get silently stuck with a keyboard catch
            %try
                var_pooled = ((ref_dat - (design(ismember(batch,ref),:)*B_hat)').^2) * ...
                    repmat(1/ref_n, ref_n, 1);
            %catch
            %    keyboard
            %end
            
            B_hat = B_hat(n_batch+1:end,:); % drop subject specific intercepts
        end
        
        function bayesdata = refCombat(obj, grand_mean, var_pooled, B_hat, dat, batch, mod, parametric, varargin)
            if any(isnan(dat(:)))
                error('Input data contains nan entries. These will break combat. Please fix and rerun');
            end

            dat = double(dat); % there are EB algorithm convergence issues if input data is single
            [sds] = std(dat')';
            wh = find(sds==0);
            [ns,~] = size(wh);
            if ns>0
                error('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')
            end
            if ischar(batch) % needed for sort function to work reliably
                batch = cellstr(batch);
            end

            meanOnly = false;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'meanOnly'
                            if varargin{i+1} == 1
                                meanOnly = true;
                            elseif varargin{i+1} == 0
                                meanOnly = false;
                            else
                                error('Did not understand argument to meanOnly');
                            end

                    end
                end
            end

            batchmod = categorical(batch);
            batchmod = dummyvar({batchmod});
            n_batch = size(batchmod,2);
            uniq_batch = unique(batch,'stable');
            fprintf('[combat] Found %d batches\n', n_batch);

            batches = cell(0);
            for i=1:n_batch
                batches{i}=find(batch == uniq_batch(i));
            end
            n_batches = cellfun(@length,batches);
            n_array = sum(n_batches);

            % Creating design matrix and removing intercept:
            if size(mod,2) ~= size(B_hat,2)
                warning('Mismatch between fit and transform design elements. Applying ComBat correction without design correction.');
                B_hat = [];
                mod = [];
            end
            
            design = [batchmod mod];
            intercept = ones(1,n_array)';
            wh = cellfun(@(x) isequal(x,intercept),num2cell(design,1));
            bad = find(wh==1);
            design(:,bad)=[];
            
            % if we have a single batch, intercept removal will remove it
            % too, so let's add it back in. This is a hacky solution
            % though. Shouldn't break anything, but there are more elegant
            % solutions available.
            if size(batchmod,2) == 1
                design = [batchmod, design];
            end

            fprintf('[combat] Adjusting for %d covariate(s) of covariate level(s)\n',size(design,2)-size(batchmod,2))

            fprintf('[combat] Standardizing Data across features\n')
            
            %Standarization Model
            stand_mean = grand_mean'*repmat(1,1,n_array);

            if not(isempty(design)) && not(isempty(B_hat))
                tmp = design(:,n_batch+1:end);
                stand_mean = stand_mean+(tmp*B_hat)';
            end	
            s_data = (dat-stand_mean)./(sqrt(var_pooled)*repmat(1,1,n_array));
            
            ignoreBatch = [];
            if ismember(obj.ref_batch_id, batch)
                sums = sum(s_data(:,batch == obj.ref_batch_id),2);
                if sum(abs(sums)) < 1e-07
                    % reference batch likely
                    ignoreBatch = obj.ref_batch_id;
                end
            end

            %Get regression batch effect parameters
            fprintf('[combat] Fitting L/S model and finding priors\n')
            batch_design = design(:,1:n_batch);
            gamma_hat = inv(batch_design'*batch_design)*batch_design'*s_data';
            delta_hat = [];
            for i=1:n_batch
                indices = batches{i};
                if meanOnly
                    delta_hat = [delta_hat; ones(1,size(s_data,1))];
                else
                    delta_hat = [delta_hat; var(s_data(:,indices)')];
                end
            end

            %Find parametric priors:
            gamma_bar = mean(gamma_hat');
            t2 = var(gamma_hat');
            delta_hat_cell = num2cell(delta_hat,2);
            a_prior=[]; b_prior=[];
            for i=1:n_batch
                a_prior=[a_prior aprior(delta_hat_cell{i})];
                b_prior=[b_prior bprior(delta_hat_cell{i})];
            end    

            if parametric
                fprintf('[combat] Finding parametric adjustments\n')
                gamma_star =[]; delta_star=[];
                for i=1:n_batch
                    if meanOnly
                        gamma_star = [gamma_star; postmean(gamma_hat(i,:),gamma_bar(i), 1, 1, t2(i))];
                        delta_star = [delta_star; ones(1,size(gamma_hat(i,:),2))];
                    else
                        indices = batches{i};
                        if ~isempty(ignoreBatch) && ignoreBatch == batch(indices(1))
                            % this is the reference batch, should only find
                            % ourselves here in fit_transform()
                            % invocations, or manually transforming the
                            % fitting sample.
                            temp = zeros(2,size(s_data,1)); % mean adjustment
                            temp(2,:) = 1; % variance scaling
                        else
                            temp = itSol(s_data(:,indices),gamma_hat(i,:),delta_hat(i,:),gamma_bar(i),t2(i),a_prior(i),b_prior(i), 0.001);
                        end
                        gamma_star = [gamma_star; temp(1,:)];
                        delta_star = [delta_star; temp(2,:)];
                    end
                end
            end

            if (1-parametric)
                gamma_star =[]; delta_star=[];
                fprintf('[combat] Finding non-parametric adjustments\n')
                for i=1:n_batch
                    if meanOnly
                        delta_hat(i,:) = 1;
                    end
                    indices = batches{i};
                    temp = inteprior(s_data(:,indices),gamma_hat(i,:),delta_hat(i,:));
                    gamma_star = [gamma_star; temp(1,:)];
                    delta_star = [delta_star; temp(2,:)];
                end
            end

            fprintf('[combat] Adjusting the Data\n')
            bayesdata = s_data;
            j = 1;
            for i=1:n_batch
                indices = batches{i};
                bayesdata(:,indices) = (bayesdata(:,indices)-(batch_design(indices,:)*gamma_star)')./(sqrt(delta_star(j,:))'*repmat(1,1,n_batches(i)));
                j = j+1;
            end
            bayesdata = (bayesdata.*(sqrt(var_pooled)*repmat(1,1,n_array)))+stand_mean;
        end
    end
end
