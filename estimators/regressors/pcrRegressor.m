classdef pcrRegressor < linearModelEstimator & modelRegressor
    properties
        numcomponents = [];
        randInt = false;
        randSlope = false;
        batch_id_funhan = @(X)((1:size(X,1))')
        fitlmeOpts = {'CovariancePattern','isotropic'};
    end
    
    properties (SetAccess = protected)                
        B = [];
        offset = 0;
        offset_null = 0;
    end
    
    properties (Access = ?baseEstimator)
        hyper_params = {'numcomponents'};
    end
    
    methods
        function obj = pcrRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
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
                            warning('Option %s not understood by pcrRegressor',varargin{i+1});
                    end
                end
            end
        end
        
        function fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            obj.offset_null = mean(Y);
            
            if obj.numcomponents == 0
                obj.B = zeros(size(X,2),1);
                obj.isFitted = true;
                obj.fitTime = toc(t0);
                return
            end
            
            % code below was copied from fmri_data/predict with minor
            % modifications for our different numenclature.
            [pc,~,~] = svd(scale(X,1)', 'econ'); % replace princomp with SVD on transpose to reduce running time. 
            %pc(:,end) = [];               % remove the last component, which is close to zero.
                                           % edit:replaced 'pc(:,size(xtrain,1)) = [];' with
                                           % end to accomodate predictor matrices with
                                           % fewer features (voxels) than trials. SG
                                           % 2017/2/6                              
                               
            % Choose number of components to save [optional]
            if ~isempty(obj.numcomponents)

                numc = obj.numcomponents;

                if obj.numcomponents > size(pc, 2)
                    warning('PCR:numcomponents','Number of components requested (%d) is more than unique components in training data (%d).', obj.numcomponents, size(pc,2));
                    numc = size(pc, 2);
                end
                pc = pc(:, 1:numc);
            end

            sc = X * pc;

            if rank(sc) == size(sc,2)
                numcomps = rank(sc); 
            elseif rank(sc) < size(sc,2)
                numcomps = rank(sc)-1;
            end

            % 3/8/13: TW:  edited to use numcomps, because sc is not always full rank during bootstrapping
            xx = [ones(size(sc, 1), 1) sc(:, 1:numcomps)];


            if ~obj.randInt && ~obj.randSlope % fit a LS or weighted LS model    
                if rank(xx) <= size(sc, 1)
                    b = pinv(xx) * Y; % use pinv to stabilize; not full rank
                    % this will only happen when bootstrapping or otherwise when there are
                    % redundant rows
                else
                    b = (xx'*xx)/xx'*Y;
                end
            else % fit random effects model
                if rank(xx) <= size(sc,2)
                    warning(['Rank deficient data found. This is not supported with mixed effects models.',...
                        'If you''re bootstrapping consider using the fixed effects approach instead']);
                end
                xxRE = [];
                if obj.randInt
                    xxRE = [xxRE, ones(size(Y,1),1)];
                end
                if obj.randSlope
                    xxRE = [xxRE, sc(:, 1:numcomps)];
                end


                batchIds = obj.batch_id_funhan(X);
                assert(length(batchIds) == length(Y),...
                    'batch_id_funhan did not return valid IDs.');
                m = fitlmematrix(double(xx), Y, double(xxRE), categorical(batchIds), obj.fitlmeOpts{:});
                b = m.fixedEffects;
            end

            % Programmers' notes: (tor)
            % These all give the same answer for full-rank design (full component)
            % X = [ones(size(sc, 1), 1) sc];
            % b1 = inv(X'*X)*X'*ytrain;
            % b2 = pinv(X)*ytrain;
            % b3 = glmfit(sc, ytrain);
            % tic, for i = 1:1000, b1 = inv(X'*X)*X'*ytrain; end, toc
            % tic, for i = 1:1000, b2 = pinv(X)*ytrain; end, toc
            % tic, for i = 1:1000, b3 = glmfit(sc, ytrain); end, toc
            % b1 is 6 x faster than b2, which is 2 x faster than b3

            obj.B = pc(:, 1:numcomps) * b(2:end);
            obj.offset = b(1);
            
            obj.isFitted = true;
            obj.fitTime = toc(t0);
        end
    end
end

