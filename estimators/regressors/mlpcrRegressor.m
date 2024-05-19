classdef mlpcrRegressor < linearModelEstimator & modelRegressor
    % multilevel PCR. Depends on mlpcr3.m, provided by canlabCore repo
    properties
        bt_dim = Inf;
        wi_dim = Inf;
        cpca = false;
        randInt = false;
        randSlope = false;
        %batch_id_funhan = @(X)((1:size(X,1))')
        batch_id_funhan = [];
        fitlmeOpts = {'CovariancePattern','isotropic'};
    end
    
    properties (SetAccess = protected)   
        Bb = [];
        Bw = [];
        
        pc_b = [];
        pc_w = [];
        
        offset = 0;
        offset_null = 0;
    end
    
    properties (Dependent = true, SetAccess = protected)
        B;
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {'bt_dim', 'wi_dim', 'cpca', 'randInt', 'randSlope'};
    end
    
    methods
        function obj = mlpcrRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'bt_dim'
                            bt_dim = varargin{i+1};
                            assert((round(bt_dim) == bt_dim && bt_dim >= 0) || isinf(bt_dim),'bt_dim must be positive integer or Inf');
                            obj.bt_dim = bt_dim;
                        case 'wi_dim'
                            wi_dim = varargin{i+1};
                            assert((round(wi_dim) == wi_dim && wi_dim >= 0) || isinf(wi_dim),'wi_dim must be positive integer or Inf');
                            obj.wi_dim = wi_dim;
                        case 'cpca'
                            cpca = varargin{i+1};
                            if ~islogical(cpca) && ~isnumeric(cpca)
                                warning('cpca should be a logical indicating a true/false value, but recieved %s type. Attempting conversion.', class(cpca));
                            end
                            obj.cpca = logical(cpca);
                        case 'randInt'
                            randInt = varargin{i+1};
                            if ~islogical(randInt) && ~isnumeric(randInt)
                                warning('cpca should be a logical indicating a true/false value, but recieved %s type. Attempting conversion.', class(randInt));
                            end
                            obj.randInt = logical(randInt);
                        case 'randSlope'
                            randSlope = varargin{i+1};
                            if ~islogical(randSlope) && ~isnumeric(randSlope)
                                warning('cpca should be a logical indicating a true/false value, but recieved %s type. Attempting conversion.', class(randSlope));
                            end
                            obj.randSlope = logical(randSlope);
                        case 'fitlmeOpts'
                            obj.fitlmeOpts = varargin{i+1};
                        case 'batch_id_funhan'
                            obj.batch_id_funhan = varargin{i+1};
                        otherwise
                            warning('Did not understand %s option', varargin{i});
                    end
                end
            end
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
            batchIds = obj.batch_id_funhan(X);
            assert(length(batchIds) == length(Y),...
                'batch_id_funhan did not return valid IDs.');
                        
            [~, bb, bw, obj.pc_b, ~, obj.pc_w] = mlpcr3(X, Y, 'subjIDs', batchIds, ...
                'numcomponents', [obj.bt_dim, obj.wi_dim], 'cpca', obj.cpca, ...
                'randInt', obj.randInt, 'randSlope', obj.randSlope, ...
                'fitlmeOpts', obj.fitlmeOpts);
            
            obj.Bb = bb(2:end);
            obj.Bw = bw(2:end);
            obj.offset = bb(1);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
        
        % for compatibility with bayesian optimizatoin categorical
        % optimizableVariable
        function set.randInt(obj, val)
            if ischar(val)
                switch(val)
                    case 'true'
                        obj.randInt = true;
                    case 'false'
                        obj.randInt = false;
                    otherwise
                        error('randInt must be true/false');
                end
            elseif categorical(true) == categorical(val)
                obj.randInt = true;
            elseif categorical(false) == categorical(val)
                obj.randInt = false;
            else
                obj.randInt = val;
            end
        end
        
        function set.randSlope(obj, val)
            if ischar(val)
                switch(val)
                    case 'true'
                        obj.randSlope = true;
                    case 'false'
                        obj.randSlope = false;
                    otherwise
                        error('randInt must be true/false');
                end
            elseif categorical(true) == categorical(val)
                obj.randSlope = true;
            elseif categorical(false) == categorical(val)
                obj.randSlope = false;
            else
                obj.randSlope = val;
            end
        end
        
        %% dependent methods
        function val = get.B(obj)
            val = obj.Bb + obj.Bw;
        end
        
        function set.B(~,~)
            error('B cannot be set directly.');
        end
    end
end

