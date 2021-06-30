% Provides the same interface as a native cvpartition matlab object, but
% allows fur N shuffleSplit partitions by wraping multiple cvpartition
% objects using 'HoldOut' partitions.
%
% Input ::
%
%   See 'help cvpartition' for most options, except for those listed below
%
% Optional Input ::
%
%   'Splits' - Followed by a scalar n indicate number of splits to use
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
classdef cvpartitionShuffleSplit
    % this method does not inherit cvpartition because cvpartition does not
    % support overlapping partitions. This is due to how it implements its
    % hidden and private obj.indices property, which occurs at a pretty 
    % low level.
    % Instead we implement something that has a very similar interface, but
    % no formal inheritance relationship with cvpartition
    properties (SetAccess = immutable)
        Type = 'ShuffleSplit';
    end
    
    properties (SetAccess = private)
        NumTestSets;
    end
    
    properties (Dependent = true)
        grp_id;
        TrainSize;
        TestSize
        NumObservations;
    end
    
    properties (Access = private, Hidden = true)
        cvpartitioners = {};
    end
    
    methods
        function obj = cvpartitionShuffleSplit(varargin)
            delete = [];
            stratify = true; % only used if stratification vector is provided as varargin{1}
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'Splits'
                            obj.NumTestSets = varargin{i+1};
                            delete = i:i+1;
                        case 'Resubstitution'
                            error('Resubstitution is not supported.');
                        case 'KFold'
                            error('KFold is not supported.');
                        case 'GroupKFold'
                            error('GroupKFold is not supported.');
                        case 'Group'
                            obj.Type = 'GroupShuffleSplit';
                        case 'Stratify'
                            stratify = varargin{i+1};
                    end
                end
            end
            varargin(delete) = [];
            
            assert(~isempty(obj.NumTestSets),...
                'You must suppy a ''Splits'' argument');
            
            for i = 1:obj.NumTestSets
                obj.cvpartitioners{i} = cvpartition2(varargin{:});
            end
            
            if length(unique(varargin{1})) > 1 && stratify % if stratifying
                % count number of entries for each unique group
                B = unique(varargin{1}(obj.cvpartitioners{1}.training()));
                n = [B, histc(varargin{1}(obj.cvpartitioners{1}.training()), B)];
                
                if length(unique(n(:,2))) > 1
                    warning(['Uneven stratification detected. This will be systematic across partitions, ', ...
                        'and should not be used with small samples. Consider changing HoldOut fraction.']);
                end
            end
        end
        
        %% dependent properties
        
        function grp_id = get.grp_id(obj)
            grp_id = obj.cvpartitioners{1}.grp_id;
        end
        
        function NumObservations = get.NumObservations(obj)
            NumObservations = obj.cvpartitioners{1}.NumObservations;
        end
        
        function n = get.TrainSize(obj)
            n = zeros(1,obj.NumTestSets);
            for i = 1:obj.NumTestSets
                n(i) = obj.cvpartitioners{i}.TrainSize;
            end
        end
        
        function n = get.TestSize(obj)
            n = zeros(1,obj.NumTestSets);
            for i = 1:obj.NumTestSets
                n(i) = obj.cvpartitioners{i}.TestSize;
            end
        end
        
        %% quasi-dependent methods
        
        function obj = repartition(obj)
            for i = 1:obj.NumTestSets
                obj.cvpartitioners{i} = obj.cvpartitioners{i}.repartition();
            end
        end
        
        function obj = set_grp_id(obj, grp_id)
            assert(strcmp(Type,'GroupShuffleSplit'),...
                'Cannot change type to GroupShuffleSplit, please create a new object instance');
            
            for i = 1:obj.NumTestSets
                obj.cvpartitioners{i} = obj.cvpartitioners{i}.set_grp_id(grp_id);
            end
        end
            
        function training = training(obj, idx)
            training = obj.cvpartitioners{idx}.training();
        end
           
        function test = test(obj, idx)
            test = obj.cvpartitioners{idx}.test();
        end     
        
        function disp(obj)
            fprintf('Shuffle Spit cross validation partition\n');
            fprintf('  NumObservatoins: %d\n', obj.NumObservations)
            fprintf('    NumTestSets: %d\n', obj.NumTestSets);
            fprintf('      TrainSize:');
            fprintf(' %d', obj.TrainSize);
            fprintf('\n      TestSize: ');
            fprintf(' %d', obj.TestSize);
            fprintf('\n');
        end
    end
end