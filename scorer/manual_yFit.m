% used when you have predicted and observed fits from somewhere and you
% want to put them into a type yfit object so that you can apply
% scorers to them.
classdef manual_yFit < yFit
    methods
        function obj = manual_yFit(Y, yfit)
            obj.Y = Y;
            obj.yfit = yfit;
        end

	function obj = set_null(obj, null_yfit)
            obj.yfit_null = null_yfit;
        end
    end
end
