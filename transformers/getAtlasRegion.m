classdef getAtlasRegion < baseTransformer    
    properties
        atlasRegion = [];
        verbose = false;
    end

    properties (Access = ?baseTransformer)
        hyper_params = {'atlasRegion'};
    end

    properties (SetAccess = private)
        atlas = [];

        isFitted = true;
        fitTime = 0;
    end
    
    methods
        function obj = getAtlasRegion(atlas, varargin)
            obj.atlas = atlas;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'atlasRegion'
                            obj.atlasRegion = varargin{i+1};
                        case 'verbose'
                            obj.verbose = varargin{i+1};
                    end
                end
            end
        end
        
        function obj = fit(obj, varargin)
            if ~isempty(obj.atlasRegion)
                obj.isFitted = true;
            else
                error('Please set obj.atlasRegion by invoking obj.set_params()');
            end
        end
        
        function dat = transform(obj, dat, varargin)
            this_region = fmri_mask_image(obj.atlas.select_atlas_subset(obj.atlasRegion));
                
            if obj.verbose, fprintf('applying atlas region mask\n'); end
            
            dat = apply_mask(dat, this_region);
        end
    end
end
