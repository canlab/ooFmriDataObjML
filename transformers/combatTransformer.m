classdef combatTransformer < fmriDataTransformer
    properties (SetAccess = private)
        combat_opts;
        get_batch_id = @(X)(X.metadata_table.subject_id);
        
        ref_batch;
        ref_batch_id;   
        
        isFitted = false;
    end
    
    methods
        function obj = combatTransformer(batch_id_funhan, combat_opts)
            if any(cellfun(@(x1)(ischar(x1) && strcmp(x1,'ref_idx')), combat_opts))
                idx = find(cellfun(@(x1)(ischar(x1) && strcmp(x1,'ref_idx')), combat_opts));
                combat_opts{idx:idx+1} = [];
                warning('Found ref_idx in combat opts, removing it...\n');
            end
            obj.combat_opts = combat_opts;
            
            obj.get_batch_id = batch_id_funhan;
        end
        
        function obj = fit(obj, dat, varargin)
            [~,~,batch_id] = unique(obj.get_batch_id(dat),'stable');
            
            % make sure dat and batch id have sensible sizes
            assert(size(dat.dat,2) == length(batch_id), 'dat be length(bach_id) x m');
            assert(ismatrix(dat.dat));
            
            obj = obj.pick_ref_batch(dat, batch_id);
            
            obj.isFitted = true;
        end
        
        function cb_dat = transform(obj, dat)
            assert(obj.isFitted,'Please call combatTransformer.fit() before plsRegressor.predict().');
            
            [~,~,batch_id] = unique(obj.get_batch_id(dat),'stable');
            
            % make sure dat and batch id have sensible sizes
            assert(size(dat.dat,2) == length(batch_id), 'dat be length(bach_id) x m');
            if isvector(dat.dat)
                idx = cellfun(@(x1)(ischar(x1) && strcmp(x1,'meanOnly')), obj.combat_opts);
                assert(~isempty(idx) && obj.combat_opts{idx+1} == 1, ...
                    'dat is a vector, but meanOnly not selected for combat.')
            else
                assert(ismatrix(dat.dat));
            end
            
            assert(size(dat.dat,1) == size(obj.ref_batch.dat,1), 'dat doesn''t match training dat dimensions');
            
            % save metadata
            cb_dat = dat;
            
            % append reference batch to list
            % re warning: var is used in evalc statement below
            dat = [dat.dat, obj.ref_batch.dat];
            
            [~, ~, batch_id] = unique(batch_id,'stable');
            tmp_ref_id = max(batch_id) + 1;
            batch_id = [batch_id(:)', tmp_ref_id*ones(1,size(obj.ref_batch.dat,2))];
            
            % apply combat
            [~,dat] = evalc('combat(dat, batch_id, obj.combat_opts{:}, ''ref'', tmp_ref_id)');
            
            % remove reference batch
            cb_dat.dat = dat(:,batch_id ~= tmp_ref_id);
        end
    end
        
    methods (Access = private)
        function obj = pick_ref_batch(obj, dat, batch_id)
            [uniq_batch_id, exp_bach_id] = unique(batch_id,'stable');

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
            sdat.dat = sdat.dat(:,exp_bach_id).^0.5; % this is the voxel-wise std for each subject

            mdat.dat = dat.dat*mumat;
            mdat.dat = mdat.dat(:,exp_bach_id); % this is voxel-wise mean for each subject

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
            
            ref_batch_idx = (obj.ref_batch_id == batch_id);
            obj.ref_batch = dat.get_wh_image(ref_batch_idx);
        end
    end
end