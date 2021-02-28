% used when you have predicted and observed fits from somewhere and you
% want to put them into a type yfit object so that you can apply
% scorers to them.
classdef manual_yFit < yFit
    methods
        function obj = manual_yFit(Y, yfit, varargin)
            obj.Y = Y;
            obj.yfit = yfit;
            if nargin > 2
                obj.yfit_raw = varargin{1};
            end
        end

        function obj = set_null(obj, null_yfit)
            obj.yfit_null = null_yfit;
        end
    end
end
