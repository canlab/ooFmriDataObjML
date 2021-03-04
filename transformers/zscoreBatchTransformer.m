% zscores each batch so that within-batch the mean value is zero and
% variance is 1
classdef zscoreBatchTransformer < Transformer
    properties (SetAccess = private)
        get_batch_id = @(X)(X.metadata_table.subject_id);
        
        isFitted = true;
        fitTime = 0;
    end
    properties (Access = ?Transformer)
        hyper_params = {};
    end
    
    methods
        function obj = zscoreBatchTransformer(batch_id_funhan)            
            obj.get_batch_id = batch_id_funhan;
        end
        
        function fit(obj, varargin)
            t0 = tic;
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function dat = transform(obj, dat)
            assert(obj.isFitted, sprintf('Please call %s.fit() before %s.transform().', class(obj), class(obj)));
            
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
                this_dat = dat.dat(:, this_idx);
                s = std(this_dat(:));
                m = mean(this_dat(:));
                dat.dat(:,this_idx) = (this_dat - m)/s; % zscore
            end
        end
    end
end
