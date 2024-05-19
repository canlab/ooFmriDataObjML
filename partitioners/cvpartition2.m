% Extension of matlab's cvpartition with extra features, mainly
% partitions that respect block dependencies (e.g. subjects).
%
% Input ::
%
%   See 'help cvpartition' for most options, except for those listed below
%
% Optional Input ::
%
%   'GroupKFold' - Followed by a scalar K indicate number of k-folds to use
%
%   'invGroupKFold' - Similar to GroupKFold, but one group is used for
%                   training and all remaining groups are used for testing.
%                   Designed for testing out of study or out of subject 
%                   generalization.
%
%   'Group'  - Followed by vector with one element per observation
%                   containing a label specifying that observations
%                   grouping. For instance if you have 5 observations from
%                   10 subjects this vector might be something like,
%                   [1,1,1,1,1,2,2,2,2,2,3,3, ...,9,9,9,9,9,10,10,10,10,10]
%
%
% Written by Bogdan Petre, sometime 2019
% Updated summer 2021
classdef cvpartition2 < cvpartition
    properties (SetAccess = protected)
        grp_id;
        invGroupKFold = false;
    end
    
    methods
        % C = cvpartition2(group, 'GroupKFold', K, 'Group', grp_id)
        function cv = cvpartition2(varargin)
            [grp_id, delete] = deal([]);
            invGroupKFold = false;
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'GroupKFold'
                            k = varargin{i+1};
                            varargin{i} = 'KFold';
                        case 'invGroupKFold'
                            invGroupKFold = true;
                            k = varargin{i+1};
                            varargin{i} = 'KFold';
                        case 'Group'
                            assert(length(varargin{i+1}) == length(varargin{1}) || varargin{1} == length(varargin{i+1}), ...
                                'grp_id must be a vector of grp_id''s of length(Group) or a scalar value equal to length(Group)');
                            grp_id = varargin{i+1};
                            [~,b] = unique(grp_id);
                            if length(varargin{1}) > 1
                                varargin{1} = varargin{1}(b);
                            else
                                varargin{1} = ones(length(b),1);
                            end
                            delete = i:i+1;
                    end
                end
            end
            varargin(delete) = [];
            
            cv@cvpartition(varargin{:});
            Impl = cvpartitionInMemoryImpl2(grp_id,invGroupKFold,varargin{:});
            Impl = Impl.updateParams();
            cv.Impl = Impl;
            
            cv.grp_id = grp_id;
            cv.invGroupKFold = invGroupKFold;
        end % cvpartition constructor
        
        function obj = set_grp_id(obj,val)
            assert(all(ismember(val,obj.grp_id)) && all(ismember(obj.grp_id,val)));
            obj.Impl = obj.Impl.set_grp_id(val);
            obj.grp_id = val;
        end
                
        
        function testidx = test(cv,varargin)
            if cv.invGroupKFold
                testidx = training(cv.Impl, varargin{:});
            else
                testidx = test(cv.Impl,varargin{:});
            end
        end
        
        function trainidx = training(cv,varargin)
            if cv.invGroupKFold
                trainidx = test(cv.Impl, varargin{:});
            else
                trainidx = training(cv.Impl,varargin{:});
            end
        end
        
        %{
        function s = get.TrainSize(cv)
            if cv.Impl.invGroupKFold
                s = cv.Impl.TestSize;
            else
                s = cv.Impl.TrainSize;
            end
        end
        
        function s = get.TestSize(cv)
            if cv.invGroupKFold
                s = cv.Impl.TrainSize;
            else
                s = cv.Impl.TestSize;
            end
        end
        %}
    end    
end