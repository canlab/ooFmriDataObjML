% this method differs from the standard one because it allows for sid
% updating
classdef cvpartitionInMemoryImpl2 < internal.stats.cvpartitionInMemoryImpl
    properties (SetAccess = private)
        sid = []
    end
    methods
        function obj = cvpartitionInMemoryImpl2(sid, varargin)
            obj = obj@internal.stats.cvpartitionInMemoryImpl(varargin{:});
            [~,~,obj.sid] = unique(sid,'stable');
        end
        
        function obj = repartition(obj, varargin)
            [~,b] = unique(obj.sid);
            obj.Group = obj.Group(b);
            obj.indices = obj.indices(b);
            obj.N = length(b);
            
            obj = repartition@internal.stats.cvpartitionInMemoryImpl(obj, varargin{:});
            
            obj = obj.updateParams();
        end
        
        function obj = set_sid(obj,val)
            [~,~,val] = unique(val,'stable');
            
            assert(all(ismember(val,obj.sid)) && all(ismember(obj.sid,val)));
            
            new_indices = zeros(size(val));
            for i = 1:length(unique(obj.indices))
                this_fold_sid = obj.sid(obj.indices == i);
                new_indices(ismember(val,this_fold_sid)) = i;
            end
            obj.indices = new_indices;
            obj.sid = val;
            obj.N = length(val);
            for i = 1:obj.NumTestSets
                obj.TrainSize(i) = sum(obj.indices ~= i);
                obj.TestSize(i) = sum(obj.indices == i);
            end
            
            uniq_sid = unique(obj.sid);
            newGroup = zeros(size(obj.sid));
            for i = 1:length(uniq_sid)
                this_sid = uniq_sid(i);
                sid_idx = this_sid == obj.sid;
                newGroup(sid_idx) = obj.Group(i);
            end
            obj.Group = newGroup;
        end
    end
    
    methods (Access = ?cvpartition2)        
        function obj = updateParams(obj)
            obj.N = length(obj.sid);
            for i = 1:obj.NumTestSets
                obj.TrainSize(i) = sum(ismember(obj.sid,find(obj.training(i))));
                obj.TestSize(i) = sum(ismember(obj.sid,find(obj.test(i))));
            end
            
            [newIndices, newGroup] = deal(zeros(obj.N,1));
            uniq_sid = unique(obj.sid);
            for i = 1:length(uniq_sid)
                this_sid = uniq_sid(i);
                sid_idx = find(this_sid == obj.sid);
                newGroup(sid_idx) = obj.Group(i);
                newIndices(sid_idx) = obj.indices(i);
            end
            obj.indices = newIndices;
            obj.Group = newGroup;
            
            if ~isempty(obj.holdoutT)
                warning(['Warning obj.holdoutT is not empty. Using this ',...
                    'function in this manner has not been tested. Please ',...
                    'check cvpartitionMemoryImpl2 and see how holdoutT is ',...
                    'being handled, compare with internal.stats.cvpartitionInMemoryImpl ',...
                    'and fixing it if necessary.']);
            end
        end
    end
end