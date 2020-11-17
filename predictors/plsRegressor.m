classdef plsRegressor < fmriDataPredictor & yFit
    properties
        numcomponents = 1;
    end
    
    properties (SetAccess = private)
        weights = fmri_data();
        offset = 0;
        
        isFitted = false;
    end
    
    properties (Access = private)
        algorithm_name = 'cv_pls';
    end
    
    methods
        function obj = plsRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
                    end
                end
            end
        end
        
        function obj = fit(obj, dat, Y)
            assert(size(dat.dat,2) == length(Y), 'length(Y) ~= size(dat.dat,2)');
            dat.Y = Y;
            
            [~,~,~,mdl] = evalc(['dat.predict(''algorithm_name'', obj.algorithm_name,', ...
                '''numcomponents'', obj.numcomponents,''nfolds'', 1)']);
            obj.weights = mdl{1};
            obj.offset = mdl{2};
            
            obj.isFitted = true;
        end
        
        function yfit = predict(obj,dat)
            assert(obj.isFitted,'Please call plsRegressor.fit() before plsRegressor.predict().');
            yfit = obj.weights(:)'*dat.dat + obj.offset;
            yfit = yfit(:);
        end
        
        function params = get_params(obj)
            params = {'numcomponents'};
        end
        
        function obj = set_hyp(obj, hyp_name, hyp_val)
            assert(strcmp(hyp_name, 'numcomponents'),...
                'Only numcomponents supported as a valid hyperparameter');
            obj.numcomponents = hyp_val;
        end
    end
end
