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
        sid;
    end
    methods
        % C = cvpartition2(group, 'GroupKFold', K, 'Group', sid)
        function cv = cvpartition2(varargin)
            [sid, delete] = deal([]);
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'GroupKFold'
                            k = varargin{i+1};
                            varargin{i} = 'KFold';
                        case 'Group'
                            assert(length(varargin{i+1}) == length(varargin{1}), ...
                                'sid must be a vector of sid''s of length(GROUP)');
                            sid = varargin{i+1};
                            [~,b] = unique(sid);
                            varargin{1} = varargin{1}(b);
                            delete = i:i+1;
                    end
                end
            end
            varargin(delete) = [];
            
            cv@cvpartition(varargin{:});
            Impl = cvpartitionInMemoryImpl2(sid,varargin{:});
            Impl = Impl.updateParams();
            cv.Impl = Impl;
            
            cv.sid = sid;
        end % cvpartition constructor
        
        function obj = set_sid(obj,val)
            assert(all(ismember(val,obj.sid)) && all(ismember(obj.sid,val)));
            obj.Impl = obj.Impl.set_sid(val);
            obj.sid = val;
        end
        
        %{
        function testidx = test(cv,varargin)
            testidx = test(cv.Impl,varargin{:});
            uniq_sid = unique(cv.sid);
            test_sid = uniq_sid(testidx);
            testidx = ismember(cv.sid,test_sid);
        end
        
        function trainidx = training(cv,varargin)
            trainidx = training(cv.Impl,varargin{:});
            uniq_sid = unique(cv.sid);
            train_sid = uniq_sid(trainidx);
            trainidx = ismember(cv.sid,train_sid);
        end
        %}
    end    
end