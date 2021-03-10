% implements canlabCore's glmfit_multilevel as a substitute for
% fmri_data/predict's cv_multilevel_glm algorithm.
%
% obj = multilevelGlmRegressor(batch_id_funhan, [options])
%
%   batch_id_funhan - handle for function that returns random effect
%                       indicators when applied to input data. e.g for
%                       feature objects with subject_id's in
%                       metadata.subject_id the following might work:
%                       @(x1)(x1.metadata.subject_id)
%
%   options ::
%
%   'glmfit_multilevel_opts' - options to pass through to
%                               glmfit_multilevel. See help glmfit_multilevel
%
%   ToDo ::
%
%   second level regressors need to be implemented. There are various ways
%   to do this, but all are ugly. baseEstimators are supposed to operate
%   on matrix data, and multilevelGlmRegressor already violates this by
%   requiring batch information. 2nd level regressors make this worse.
%
%   'get_2ndlvl_reg_funhan' - handle for function that returns second level
%                       regressors for X. One value per row/entry of X 
%                       (with redundant entries across repeated entries for
%                       the same block). If your regressor is age and you
%                       have your data sorted by subject with 4 entries per
%                       subject multilevelGlmRegressor will expect
%                       something like this to be returned by
%                       get_2ndlvl_reg_funhan,
%                       [36 36 36 36 25 25 25 25 48 48 48 48 ... ]'
%                       You might retrieve this from an age field in an
%                       fmri_data.metadata_table.age entry like so,
%                       @(x1)(x1.metadata_table.age)
%                       Note: Model is fit with second level regressors,
%                       but model prediction ignores them. Implementation 
%                       is here but untested, so it's been disabled in the
%                       constructor (see commented code).
%  
classdef multilevelGlmRegressor < linearModelEstimator & modelRegressor
   properties
        numcomponents = 1;
    end
    
    properties (SetAccess = protected)   
        get_batch_id = @(X)(X.metadata.subject_id);
        get_secondlvl_reg = [];

        isFitted = false;
        fitTime = -1;
        
        B = [];
        offset = 0;
        offset_null = 0;
        
        glmfit_opts = {};
        
        stats = [];
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {};
    end
    
    methods
        function obj = multilevelGlmRegressor(batch_id_funhan, varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'glmfit_multilevel_opts'
                            obj.glmfit_opts = varargin{i+1};
                            
                        %case 'get_2ndlvl_reg_funhan'
                        %    obj.get_secondlvl_reg = varargin{i+1};
                    end
                end
            end
            
            obj.get_batch_id = batch_id_funhan;
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
            [~,exemplar,batch_ids] = unique(obj.get_batch_id(X),'stable');
            if ~isempty(obj.get_secondlvl_reg)
                warning('Model will be fit while controlling for specified second level regressors, but predict doesn''t support them yet.');
            
                Xsl = obj.get_secondlvl_reg(X);
                Xsl = Xsl(exemplar,:);
            else
                Xsl = [];
            end
            
            [Xml, Yml] = deal(cell(size(unique(batch_ids))));
            
            diffs = diff(batch_ids);

            % build cell array of X and Y, using a varargin input identifying subjects.
            count = 1;
            for i=1:numel(Y)
                Xml{count}(end+1,:) = X(i,:);
                Yml{count}(end+1,:) = Y(i);

                if i~=numel(Y) && diffs(i) >= 1, count = count+1; end % on to a new subject.  all of a subject's trials must be adjacent to each other.
            end
            
            % call ml glm but suppress output
            obj.stats = glmfit_multilevel(Yml, Xml, Xsl, obj.glmfit_opts{:});
            
            obj.offset = obj.stats.beta(1,1);
            obj.B = obj.stats.beta(1,2:end)';
                        
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end
