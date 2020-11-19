% zscores each batch so that within-batch all voxels have mean zero and
% variance 1.
classdef zscoreVxlTransformer < fmriDataTransformer
    properties (SetAccess = private)
        get_batch_id = @(X)(X.metadata_table.subject_id);
        
        isFitted = true;
        fitTime = 0;
    end
    
    methods
        function obj = zscoreVxlTransformer(batch_id_funhan)            
            obj.get_batch_id = batch_id_funhan;
        end
        
        function obj = fit(obj, varargin)
            t0 = tic;
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function dat = transform(obj, dat)
            assert(obj.isFitted,'Please call zscoreVxlTransformer.fit() before zscoreVxlTransformer.transform().');
            
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
            
            uniq_batch_id = unique(batch_id,'stable');
            n_batch_id = length(uniq_batch_id);
            for i = 1:n_batch_id
                this_batch = uniq_batch_id(i);
                this_idx = find(this_batch == batch_id);
                dat.dat(:,this_idx) = zscore(dat.dat(:, this_idx),[],2); % zscore row-wise
            end
        end
    end
end