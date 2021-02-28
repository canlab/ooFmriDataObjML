classdef pcrRegressor < linearModelEstimator & modelRegressor
    properties
        numcomponents = [];
    end
    
    properties (SetAccess = private)                
        isFitted = false;
        fitTime = -1;
    end
    
    properties (Access = ?Estimator)
        hyper_params = {'numcomponents'};
    end
    
    methods
        function obj = pcrRegressor(varargin)
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch(varargin{i})
                        case 'numcomponents'
                            obj.numcomponents = varargin{i+1};
                    end
                end
            end
        end
        
        function obj = fit(obj, X, Y)
            t0 = tic;
            assert(size(X,1) == length(Y), 'length(Y) ~= size(X, 1)');
            
            % code below was copied from fmri_data/predict with minor
            % modifications for our different numenclature.
            [pc,~,~] = svd(scale(X,1)', 'econ'); % replace princomp with SVD on transpose to reduce running time. 
            pc(:,end) = [];                % remove the last component, which is close to zero.
                                           % edit:replaced 'pc(:,size(xtrain,1)) = [];' with
                                           % end to accomodate predictor matrices with
                                           % fewer features (voxels) than trials. SG
                                           % 2017/2/6                              
                               
            % Choose number of components to save [optional]
            if ~isempty(obj.numcomponents)

                numc = obj.numcomponents;

                if obj.numcomponents > size(pc, 2)
                    disp('WARNING!! Number of components requested is more than unique components in training data.');
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
            X = [ones(size(sc, 1), 1) sc(:, 1:numcomps)];

            if rank(X) <= size(sc, 1)
                b = pinv(X) * Y; % use pinv to stabilize; not full rank
                % this will only happen when bootstrapping or otherwise when there are
                % redundant rows
            else
                b = inv(X'*X)*X'*Y;
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

