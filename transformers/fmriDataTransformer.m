classdef fmriDataTransformer < Transformer
    properties (SetAccess = protected)
        fitTransformTime = -1;
    end

    properties (SetAccess = private)
       transformer = [];
    end    

    properties (Abstract, Access = ?Transformer)
        hyper_params;
    end
    
    methods
	function obj = fmriDataTransformer(transformer)
            assert(isa(transformer,'Transformer"), sprintf('transformer must be type Transformer, but type %s found', class(transformer)));
            obj.transformer = transformer;
	end

        function [obj, dat] = fit_transform(obj, varargin)
            t0 = tic;
            obj = obj.fit(varargin{:});
            dat = obj.transform(varargin{:});
            obj.fitTransformTime = toc(t0);
        end

        function obj = fit(obj, dat, varargin)
            obj.transformer = obj.transformer.fit(dat.dat', varargin{:});
        end

        function dat = transform(obj, dat, varargin)
            dat = obj.transformer.transform(dat', varargin{:});
        end
        
        function params = get_params(obj)
            params = obj.transformer.hyper_params;
        end
        
        % if a estimator has hyperparameters, this sets them. 
        function obj = set_hyp(obj, hyp_name, hyp_val)
            params = obj.transformer.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj.transformer)));
            
            obj.transformer.set_hyp(hyp_name, hyp_val);
        end
    end
end
