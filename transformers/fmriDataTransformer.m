classdef fmriDataTransformer
    properties (SetAccess = protected)
        fitTransformTime = -1;
    end
    methods (Abstract)
        fit(obj)
        transform(obj)
    end
    methods
        function [obj, dat] = fit_transform(obj, varargin)
            t0 = tic;
            obj = obj.fit(varargin{:});
            dat = obj.transform(varargin{:});
            obj.fitTransformTime = toc(t0);
        end
    end
end
