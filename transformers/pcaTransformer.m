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
    properties(Dependent = true)
        numcomponents
    end
    properties (Access = ?baseTransformer)
       	hyper_params = {'numcomponents'};
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

            iscentered = false;
            for i = 1:length(obj.pca_args)
                if ischar(obj.pca_args{i}) && strcmp(obj.pca_args{i}, 'Centered')
                    iscentered = obj.pca_args{i+1};
                end
            end
            if iscentered
                dat = (dat - mean(dat));
            else
                dat = (dat - mean(dat));
            end

            dat = dat*obj.coeffs;
        end

        function set.numcomponents(obj,n)
            % if NumComponents is specified already, remove it and its argument
            nc_ind = [];
            for i = 1:length(obj.pca_args)
                if ischar(obj.pca_args{i}) && strcmp(obj.pca_args{i}, 'NumComponents')
                    nc_ind = i;
                end
            end
            if ~isempty(nc_ind)
                obj.pca_args{[nc_ind, nc_ind+1]} = [];
            end

            % assign new component and argument to params
            obj.pca_args = [obj.pca_args, 'NumComponents', n];
        end

        function n = get.numcomponents(obj)
            n = [];
            
            for i = 1:length(obj.pca_args)
                if ischar(obj.pca_args{i}) && strcmp(obj.pca_args{i}, 'NumComponents')
                    n = obj.pca_args{i+1};
                    break;
                end
            end
        end
    end
end
