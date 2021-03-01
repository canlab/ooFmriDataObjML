classdef (Abstract) crossValidator < yFit
    properties
        repartOnFit = true;
        cv = @(dat, Y)cvpartition2(ones(length(dat.Y),1),'KFOLD', 5, 'Stratify', dat.metadata_table.subject_id);
        n_parallel = 1;
        
        estimator = [];
    end
    
    properties (SetAccess = protected)
        fold_lbls = [];
        cvpart = [];
        foldEstimator = {};
        
        verbose = true;
        
        evalTime = -1;
        is_done = false;
    end
    
    methods (Abstract)
        do(obj)
    end
    
    methods
        function obj = crossValidator(estimator, cv, varargin)            
            assert(isa(estimator,'Estimator'), 'estimator must be type Estimator');
            
            obj.estimator = estimator;
            if ~isempty(cv), obj.cv = cv; end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'repartOnFit'
                            obj.repartOnFit = varargin{i+1};
                        case 'n_parallel'
                            obj.n_parallel = varargin{i+1};
                        case 'verbose'
                            obj.verbose = varargin{i+1};
                    end
                end
            end
        end    
        
        %% converter objects 
        % converts between predict and score versions. This is dirty, but
        % can potentially save a ton of time. Best not to incorporate these
        % into the internal logic of any other classes, but may be useful
        % nonetheless on the command line interface or applications of 
        % existing classes in scripts.
        function obj = crossValPredict(cvObj)
            assert(isa(cvObj, 'crossValScore'), 'Only convesion of crossValScore to crossValPredict is supported at this time.');
            
            this_baseEstimator = getBaseEstimator(cvObj.estimator);
            if isa(this_baseEstimator, 'modelRegressor') % scores are predictions, makes things easy
                obj = crossValPredict(cvObj.estimator, cvObj.cv);
                
                fnames_score = fieldnames(cvObj);
                fnames_predict = fieldnames(obj);
                for i = 1:length(fnames_predict)
                    if ismember(fnames_predict{i}, fnames_score)
                        try
                            obj.(fnames_predict{i}) = cvObj.(fnames_predict{i});
                        catch
                            warning('Dropping %s.',fnames_predict{i});
                        end
                    end
                end        
                
                obj.yfit = obj.yfit_raw;
            elseif isa(this_baseEstimator,'modelClf')
                warning('crossValidator:crossValPredict','You will not be able to convert this object back to crossValScore due to information loss');
                obj = crossValPredict(cvObj.estimator, cvObj.cv);
                
                fnames_score = fieldnames(cvObj);
                fnames_predict = fieldnames(obj);
                for i = 1:length(fnames_predict)
                    if ismember(fnames_predict{i}, fnames_score)
                        try
                            obj.(fnames_predict{i}) = cvObj.(fnames_predict{i});
                        catch
                            warning('Dropping %s.',fnames_predict{i});
                        end
                    end
                end        
                
                obj.yfit = this_baseEstimator.decisionFcn(cvObj.yfit_raw);
            else 
                error('Conversion of %s to crossValPredict is not supported', ...
                    class(cvObj));
            end
        end
        
        
        function obj = crossValScore(cvObj, scorer)     
            assert(isa(cvObj, 'crossValPredict'), 'Only convesion of crossValPredict to crossValScore is supported at this time.');
            
            this_baseEstimator = getBaseEstimator(cvObj.estimator);
            if isa(this_baseEstimator, 'modelRegressor')
                obj = crossValScore(cvObj.estimator, cvObj.cv, scorer);
                
                fnames_predict = fieldnames(cvObj);
                fnames_score = fieldnames(obj);
                for i = 1:length(fnames_predict)
                    if ismember(fnames_predict{i}, fnames_score)
                        try
                            obj.(fnames_predict{i}) = cvObj.(fnames_predict{i});
                        catch
                            warning('Dropping %s.',fnames_predict{i});
                        end
                    end
                end        
                obj.yfit_raw = cvObj.yfit;
                obj = obj.eval_score();
            elseif isa(this_baseEstimator, 'modelClf')
                error('crossValPredict objects with modelClf as their base estimators cannot be converted to crossValScore.');
            else
                error('Conversion of %s to crossValPredict is not supported', ...
                    class(cvObj));
            end
        end
    end
end