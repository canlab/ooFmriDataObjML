% this method differs from the standard one because it allows for grp_id
% updating
classdef cvpartitionInMemoryImpl2 < internal.stats.cvpartitionInMemoryImpl
    properties (SetAccess = private)
        grp_id = []
    end
    methods
        function obj = cvpartitionInMemoryImpl2(grp_id, varargin)
            obj = obj@internal.stats.cvpartitionInMemoryImpl(varargin{:});
            [~,~,obj.grp_id] = unique(grp_id,'stable');
        end
        
        function obj = repartition(obj, varargin)
            if ~isempty(obj.grp_id)
                [~,b] = unique(obj.grp_id);
                obj.Group = obj.Group(b);
                obj.indices = obj.indices(b);
                obj.N = length(b);
            end
            
            obj = repartition@internal.stats.cvpartitionInMemoryImpl(obj, varargin{:});
            
            obj = obj.updateParams();
        end
        
        function obj = set_grp_id(obj,val)
            [~,~,val] = unique(val,'stable');
            
            assert(all(ismember(val,obj.grp_id)) && all(ismember(obj.grp_id,val)));
            
            new_indices = zeros(size(val));
            for i = 1:length(unique(obj.indices))
                this_fold_grp_id = obj.grp_id(obj.indices == i);
                new_indices(ismember(val,this_fold_grp_id)) = i;
            end
            obj.indices = new_indices;
            obj.grp_id = val;
            obj.N = length(val);
            for i = 1:obj.NumTestSets
                obj.TrainSize(i) = sum(obj.indices ~= i);
                obj.TestSize(i) = sum(obj.indices == i);
            end
            
            uniq_grp_id = unique(obj.grp_id);
            newGroup = zeros(size(obj.grp_id));
            for i = 1:length(uniq_grp_id)
                this_grp_id = uniq_grp_id(i);
                grp_id_idx = this_grp_id == obj.grp_id;
                newGroup(grp_id_idx) = obj.Group(i);
            end
            obj.Group = newGroup;
        end
    end
    
    methods (Access = ?cvpartition2)        
        function obj = updateParams(obj)
            if ~isempty(obj.grp_id)
                obj.N = length(obj.grp_id);
                for i = 1:obj.NumTestSets
                    obj.TrainSize(i) = sum(ismember(obj.grp_id,find(obj.training(i))));
                    obj.TestSize(i) = sum(ismember(obj.grp_id,find(obj.test(i))));
                end
                [newIndices, newGroup] = deal(zeros(obj.N,1));
                uniq_grp_id = unique(obj.grp_id);
                for i = 1:length(uniq_grp_id)
                    this_grp_id = uniq_grp_id(i);
                    grp_id_idx = find(this_grp_id == obj.grp_id);
                    newGroup(grp_id_idx) = obj.Group(i);
                    newIndices(grp_id_idx) = obj.indices(i);
                end
                obj.indices = newIndices;
                obj.Group = newGroup;

                if ~isempty(obj.holdoutT)
                    warning('cvpartitionMemoryImpl2:updateParams',['Warning obj.holdoutT is not empty. Using this ',...
                        'function in this manner has not been tested. Please ',...
                        'check cvpartitionMemoryImpl2 and see how holdoutT is ',...
                        'being handled, compare with internal.stats.cvpartitionInMemoryImpl ',...
                        'and fixing it if necessary.']);
                end
            end
        end
    end
end