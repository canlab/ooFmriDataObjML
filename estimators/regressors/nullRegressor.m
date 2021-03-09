% dummy regressor, computes the mean of whatever its input.
% there aren't many things you need this for, but it can be
% a convenient drop in at times if you want to replicate your
% model interface only with a null model, say to pass it off
% to a scorer.
classdef nullRegressor < linearModelEstimator & modelRegressor    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
        
        B = [];
        offset = 0;
        offset_null = 0;
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {};
    end
    
    methods
        function obj = olsRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'intercept'
                            obj.intercept = varargin{i+1};
                    end
                end
            end
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            [obj.offset, obj.offset_null] = deal(mean(Y));
            
            obj.B = zeros(size(X,2),1);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end
