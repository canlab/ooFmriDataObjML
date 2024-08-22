classdef pcaTransformer < baseTransformer
    properties
        pca_args = {};
    end
    properties (SetAccess = private)
        isFitted = true;
        fitTime = 0;
    end
    properties(SetAccess = protected)
        coeffs = [];
    end
    properties (Access = ?baseTransformer)
       	hyper_params = {};
    end
    
    methods
        function obj = pcaTransformer(varargin)            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'pca_args'
                            obj.pca_args = varargin{i+1};
                        otherwise
                            warning('Did not understand argument %s', varargin{i});
                    end
                end
            end
        end
        
        function fit(obj, dat, varargin)
            % assumes each row is an observation
            t0 = tic;
            [obj.coeffs, ~] = pca(double(dat), obj.pca_args{:}, 'Centered', false);
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function dat = transform(obj, dat)
            assert(obj.isFitted,'Please call pcaTransformer.fit() before pcaTransformer.transform().');
            
            if ismember('Centered', obj.pca_args)
                if obj.pca_args{find(ismember(obj.pca_args,'Centered')) + 1}
                    dat = (dat - mean(dat));
                end
            else
                dat = (dat - mean(dat));
            end

            dat = dat*obj.coeffs;
        end
    end
end
