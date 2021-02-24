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
        
        is_done = false;
    end
    
    methods (Abstract)
        do(obj)
        
        do_null(obj);
    end
end