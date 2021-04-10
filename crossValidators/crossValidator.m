classdef (Abstract) crossValidator < yFit
    properties
        repartOnFit = true;
        cv = @(dat, Y)cvpartition(ones(length(dat.Y),1),'KFOLD', 5);
        n_parallel = 1;
        
        estimator = [];
    end
    
    properties (SetAccess = protected)
        foldEstimator = {};
        
        verbose = true;
        
        evalTime = -1;
        is_done = false;
    end
    
    properties (Dependent = true)
        fold_lbls;
        classLabels;
    end
    
    properties (Abstract)
        cvpart;
    end
    
    methods (Abstract)
        do(obj)
    end
    
    methods
        function obj = crossValidator(estimator, cv, varargin)            
            assert(isa(estimator,'baseEstimator'), 'estimator must be type baseEstimator');
            
            obj.estimator = copy(estimator);
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
            
            obj = crossValPredict(copy(cvObj.estimator), cvObj.cv); 

            fnames_score = fieldnames(cvObj);
            fnames_predict = fieldnames(obj);
            for i = 1:length(fnames_score)
                % don't try to copy classLabels because it's a non-setable
                % property, doesn't need to be set though as long as we
                % copy Y.
                if ismember(fnames_score{i}, fnames_predict) && ~strcmp(fnames_score{i}, 'classLabels')
                    try
                        if isa(cvObj.(fnames_score{i}), 'matlab.mixin.Copyable')
                            obj.(fnames_score{i}) = copy(cvObj.(fnames_score{i}));
                        else
                            if isa(cvObj.(fnames_score{i}),'handle')
                                warning('Copying obj.%s by reference!', fnames_score{i});
                            end
                            obj.(fnames_score{i}) = cvObj.(fnames_score{i});
                        end
                    catch
                        warning('Dropping %s.',fnames_score{i});
                    end
                end
            end

            % conversion to crossValPredict must be done on a fold by fold
            % basis to handle conversion of classification scores, but
            % we'll do it for both for consistency
            obj.yfit = [];
            for i = 1:cvObj.cvpart.NumTestSets
                fold_idx = cvObj.cvpart.test(i);
                this_baseEstimator = getBaseEstimator(cvObj.foldEstimator{i});
                if  isa(this_baseEstimator,'modelClf')
                    warning('crossValidator:crossValPredict','You will not be able to convert this object back to crossValScore due to information loss');
                    
                    % we need to reorder raw scores to match base
                    % estimator's order, which isn't guaranteed to match
                    assert(length(this_baseEstimator.classLabels) == length(obj.classLabels), ...
                        'Number of class labels in Y don''t match number in Y partition. Check that cvpartitioner is appropriately straifying outcomes cross folds.');
                    nClasses = length(obj.classLabels);
                    if nClasses > 2
                        resortIdx = zeros(1,nClasses);
                        for j = 1:nClasses % the assertion above ensures fold estimators will have nClasses
                            resortIdx(j) = find(this_baseEstimator.classLabels(j) == obj.classLabels);
                        end
                    else
                        assert(size(cvObj.yfit_raw,2) == 1, sprintf('Binary classifiers should only have one score per observation but %d found.', size(cvObj.yfit_raw,2)));
                        resortIdx = 1;
                    end
                        
                    obj.yfit = [obj.yfit; this_baseEstimator.decisionFcn(cvObj.yfit_raw(fold_idx, resortIdx))];
                elseif isa(this_baseEstimator, 'modelRegressor')
                    obj.yfit = [obj.yfit; obj.yfit_raw(fold_idx)];
                else
                    error('Conversion of %s to crossValPredict is not supported', ...
                        class(cvObj));
                end
            end
            
            % yfit is currently sorted by fold, let's fix that by figuring
            % out what the fold sorting is and reversing it.
            [~,I] = sort(obj.fold_lbls);
            [~,II] = sort(I);
            obj.yfit = obj.yfit(II);
        end
        
        
        function obj = crossValScore(cvObj, scorer)     
            assert(isa(cvObj, 'crossValPredict'), 'Only convesion of crossValPredict to crossValScore is supported at this time.');
            
            this_baseEstimator = getBaseEstimator(cvObj.estimator);
            if isa(this_baseEstimator, 'modelRegressor')
                obj = crossValScore(copy(cvObj.estimator), cvObj.cv, scorer);
                
                fnames_predict = fieldnames(cvObj);
                fnames_score = fieldnames(obj);
                for i = 1:length(fnames_predict)
                    % don't try to copy classLabels because it's a non-setable
                    % property, doesn't need to be set though as long as we
                    % copy Y.
                    if ismember(fnames_predict{i}, fnames_score) && ~strcmp(fnames_predict{i},'classLabels')
                        try
                            if isa(cvObj.(fnames_predict{i}), 'matlab.mixin.Copyable')
                                obj.(fnames_predict{i}) = copy(cvObj.(fnames_predict{i}));
                            else
                                if isa(cvObj.(fnames_predict{i}),'handle')
                                    warning('Copying obj.%s by reference!', fnames_predict{i});
                                end
                                obj.(fnames_predict{i}) = cvObj.(fnames_predict{i});
                            end
                        catch
                            warning('Dropping %s.',fnames_predict{i});
                        end
                    end
                end        
                obj.yfit_raw = cvObj.yfit(:);
                obj.eval_score();
            elseif isa(this_baseEstimator, 'modelClf')
                error('crossValPredict objects with modelClf as their base estimators cannot be converted to crossValScore.');
            else
                error('Conversion of %s to crossValPredict is not supported', ...
                    class(cvObj));
            end
        end
        
        %% dependent properties
        function val = get.classLabels(obj)
            val = unique(obj.Y, 'stable');
        end
        
        function set.classLabels(~, ~)
            error('Class labels cannot be set explicitly, they''re determined by unique entries in obj.Y');
        end
        
        function val = get.fold_lbls(obj)
            val = zeros(length(obj.Y),1);
            for i = 1:obj.cvpart.NumTestSets
                assert(all(val(obj.cvpart.test(i)) == 0),...
                    'Fold partitions have nonzero intersecting set: test value assigned to multiple folds. Please query fold membership from obj.cvpart directly.');
                
                val(obj.cvpart.test(i)) = i;
            end
        end
        
        function set.fold_lbls(~,~)
            error('You cannot set fold_lbls directly. They are set by elements of obj.cvpart');
        end                    
    end
    
    
    
    methods (Access = protected)
        function newObj = copyElement(obj)
            newObj = copyElement@matlab.mixin.Copyable(obj);
            
            fnames = fieldnames(obj);
            newObj.foldEstimator = copyCell(obj.foldEstimator);
            fnames(ismember(fnames,'foldEstimator')) = [];
            
            for i = 1:length(fnames)
                if isa(obj.(fnames{i}), 'cell')
                    hasHandles = checkCellsForHandles(obj.(fnames{i}));
                    if hasHandles
                        try
                            newObj.(fnames{i}) = cell(size(obj.(fnames{i})));
                            warning('%s.%s has handle objects, but corresponding deep copy support hasn''t been implemented. Dropping %s.',class(obj), fnames{i}, fnames{i});
                        catch
                            error('%s.%s has handle objects, corresponding copy support hasn''t been implemented, and element cannot be dropped. Cannot complete deep copy.',class(obj), fnames{i});
                        end
                    end
                elseif isa(obj.(fnames{i}), 'matlab.mixin.Copyable')
                    newObj.(fnames{i}) = copy(obj.(fnames{i}));
                elseif isa(obj.(fnames{i}), 'handle') % implicitly: & ~isa(obj.(fnames{i}), 'matlab.mixin.Copyable')
                    % the issue here is that fuction handles that are
                    % copied can contain references to the object they
                    % belong to, but these references will continue to
                    % point to the original object, and not the copy
                    % becaues matlab cannot parse these function handles
                    % appropriately.
                    warning('%s.%s is a handle but not copyable. This can lead to unepected behavior and is not ideal', class(obj), fnames{i});
                end
            end
        end
    end
end
