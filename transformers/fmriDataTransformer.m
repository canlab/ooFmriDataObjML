classdef fmriDataTransformer
    methods (Abstract)
        fit(obj)
        transform(obj)
    end
    methods
        function [obj, dat] = fit_transform(obj, varargin)
            obj = obj.fit(varargin{:});
            dat = obj.transform(varargin{:});
        end
    end
end
