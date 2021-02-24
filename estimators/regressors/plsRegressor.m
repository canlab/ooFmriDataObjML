classdef plsRegressor < fmriDataRegressor & yFit
    properties
        numcomponents = 1;
    end
    
    properties (SetAccess = private)
        algorithm_name = 'cv_pls';
                
        isFitted = false;
        fitTime = -1;
    end
    
    properties (Access = ?fmriDataEstimator)
        hyper_params = {'numcomponents'};
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
            t0 = tic;
            assert(size(dat.dat,2) == length(Y), 'length(Y) ~= size(dat.dat,2)');
            dat.Y = Y;
            
            [~,~,~,mdl] = evalc(['dat.predict(''algorithm_name'', obj.algorithm_name,', ...
                '''numcomponents'', obj.numcomponents,''nfolds'', 1)']);
            
	    % get prototype image
            % this needs a solution for when image 1 differs in voxel count
            % from the full dataset
            obj.weights = fmri_data(dat.get_wh_image(1));
            fnames = {'images_per_session', 'Y', 'Y_names', 'Y_descrip', 'covariates',...
                'additional_info', 'metadata_table', 'dat_descrip', 'image_names', 'fullpath'};
            for field = fnames
                obj.weights.(field{1}) = [];
            end

            obj.weights.dat = mdl{1}(:);
            obj.weights.history = {[class(obj), ' fit']};
            
            obj.offset = mdl{2};
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
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
            if fast && size(obj.weights.dat,1) == size(dat.dat,1)
                yfit = obj.weights.dat(:)'*dat.dat + obj.offset;
            else
                yfit = apply_mask(dat, obj.weights, 'pattern_expression', 'dotproduct', 'none') + obj.offset;
            end
            yfit = yfit(:);
        end
    end
end

