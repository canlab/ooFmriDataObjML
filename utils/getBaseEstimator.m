function estimator = getBaseEstimator(obj)
    if isa(obj,'Estimator')
        estimator = obj;
        
        fnames = fieldnames(obj);
        for i = 1:length(fnames)
            if isa(obj.(fnames{i}),'Estimator')
                estimator = getBaseEstimator(obj.(fnames{i}));
                break;
            end
        end
    else
        error('getBaseEstimator(obj) expects obj to be of type ''Estimator''');
    end
end