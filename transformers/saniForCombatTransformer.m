classdef saniForCombatTransformer < fmriDataTransformer
    properties (SetAccess = private)
        get_batch_id = @(X)(X.metadata_table.subject_id);
    end
    
    methods
        function obj = saniForCombatTransformer(batch_id_funhan)
            obj.get_batch_id = batch_id_funhan;
        end
        
        function obj = fit(obj, varargin)
        end
        
        function clean_dat = transform(obj, dat)
            batch_id = obj.get_batch_id(dat);
            clean_dat = sani_for_combat(dat, batch_id);
        end
    end
end