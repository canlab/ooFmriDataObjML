classdef baseTransformer < handle & matlab.mixin.Copyable
    % handle inheritance is required because pipelines have to inherit both
    % Transformers and Estimators, and Estimators must be able to have
    % dynamic properties for multiclass classifiers (which need unique
    % properties for every classifier instance)
    properties (SetAccess = protected)
        fitTransformTime = -1;
    end
    properties (Abstract, Access = ?baseTransformer)
        hyper_params;
    end
    methods (Abstract)
        fit(obj)
        transform(obj)
    end
    methods
        function dat = fit_transform(obj, varargin)
            t0 = tic;
            obj.fit(varargin{:});
            dat = obj.transform(varargin{:});
            obj.fitTransformTime = toc(t0);
        end
        
        function params = get_params(obj)
            params = obj.hyper_params;
        end
        
        % if a estimator has hyperparameters, this sets them. 
        function set_params(obj, hyp_name, hyp_val)
            params = obj.get_params();
            assert(ismember(hyp_name, params), ...
                sprintf('%s is not a hyperparameter of %s\n', hyp_name, class(obj)));
            
            obj.(hyp_name) = hyp_val;
        end
    end
    
    methods (Access = protected)
        function newObj = copyElement(obj)
            newObj = copyElement@matlab.mixin.Copyable(obj);
            
            fnames = fieldnames(obj);
            for i = 1:length(fnames)
                if isa(obj.(fnames{i}), 'cell')
                    hasHandles = checkCellsForHandles(obj.(fnames{i}));
                    if hasHandles
                        try
                            newObj.(fnames{i}) = cell(size(obj.(fnames{i})));
                            warning('%s.(%s) has handle objects, but corresponding deep copy support hasn''t been implemented. Dropping %s.',class(obj), fnames{i}, fnames{i});
                        catch
                            error('%s.(%s) has handle objects, corresponding copy support hasn''t been implemented, and element cannot be dropped. Cannot complete deep copy.',class(obj), fnames{i});
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
