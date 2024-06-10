% z-scores features (column mean = 0, std=1)
classdef standardScalar < baseTransformer
    properties (SetAccess = private)
        isFitted = true;
        fitTime = 0;

        mu = [];
        std = [];
    end
    properties (Access = ?baseTransformer)
        hyper_params = {};
    end
    
    methods
        function obj = standardScalar(varargin)
        end
        
        function fit(obj, dat, varargin)
            % assumes each row is an observation
            t0 = tic;
            obj.mu = nanmean(dat,1);
            obj.std = nanstd(dat,1,1); % normalize by N (not N-1) columnwise

            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function dat = transform(obj, dat)
            assert(obj.isFitted,'Please call pcaTransformer.fit() before pcaTransformer.transform().');
            
            dat = (dat - obj.mu)./obj.std;

            % these rows are essentially invariant within singleton precision and blow up.
            bad_rows = obj.std == 0 | abs(obj.std./obj.mu) < 1e-7;
            dat(:,bad_rows) = 0;
        end
    end
end
