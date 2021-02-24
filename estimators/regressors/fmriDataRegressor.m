classdef (Abstract) fmriDataRegressor < fmriDataEstimator
% you could make this private like so, and you should consider this option.
%   properties (SetAccess = {?crossValidator, ?fmriDataTransformer, ?fmriDataEstimator})
    properties
        weights = fmri_data();
        offset = 0;
    end

    methods        
        function yfit = predict(obj, dat, varargin)
            fast = false;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'fast'
                            fast = varargin{i+1};
                    end
                end
            end
                            
            
            assert(obj.isFitted,sprintf('Please call %s.fit() before %s.predict().\n',class(obj)));
            if fast
                yfit = obj.weights(:)'*dat.dat + obj.offset;
            else
                yfit = apply_mask(dat, obj.weights, 'pattern_expression', 'dotproduct', 'none') + obj.offset;
            end
            yfit = yfit(:);
        end
        
    end
end
