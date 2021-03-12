% transformer = fmriData2RegionAvgs(atlas, [options])
% converts an image_vector object into a vector of region averages,
% returned as a features datatype.
%
% options ::
%   metadataConstructor_funhan - a function that is run on input data to
%       extract metadata to bundle with the output features object. e.g.
%       @(X)(X.metadata_table.subject_id) results in a features object with
%       the region averages in it's .dat field and with subject_id info in
%       its .metadata field
% 
% you could do this with a function transformer too, but I don't know of
% any way to do it that exposes the atlas used to the user afterwards, and
% there might be resampling issues down the line when trying to use the
% model on novel data that may or may not be in the same space as the
% original.
classdef fmriData2RegionAvgs < baseTransformer
    % applies an arbitrary function to the data.
    properties (SetAccess = private)
        metadataConstructor = @(X)(features(X.dat'));

        isFitted = true;
        fitTime = 0;
    end
    
    properties
        atlas = []
        regionSizeThreshold = 0;
    end
    
    properties (Access = ?baseTransformer)
        hyper_params = {};
    end
    
    methods
        function obj = fmriData2RegionAvgs(atlas, varargin)            
             for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'metadataConstructor_funhan'
                            obj.metadataConstructor = varargin{i+1};
                        case 'regionSizeThreshold'
                            obj.regionSizeThreshold = varargin{i+1};
                        otherwise
                            warning('Did not understand input %s to fmri2VxlFeatTransformer', varargin{i});
                    end
                end
             end
            
            obj.atlas = atlas;
        end
        
        function fit(obj, X, varargin)
            t0 = tic;
            
            assert(isa(X,'image_vector'), sprintf('Input must be type image_vector but received type %s.',class(X)));
                        
            obj.atlas = obj.atlas.resample_space(X);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        function X = transform(obj, X)
            assert(obj.isFitted,sprintf('Please call %s.fit() before %s.transform().', class(obj), class(obj)));
            
            metadata = obj.metadataConstructor(X);
            
            % extract_roi_averages will resample the atlas to the brain,
            % but we don't want to change the space when testing new data, 
            % so let's make sure X is already in the atlas space so that
            % extract_roi_averages' resampling doesn't break things.
            X = X.resample_space(obj.atlas); 
            
            X = features(cell2mat({X.extract_roi_averages(obj.atlas).dat}),...
                metadata);
        end
    end
end
