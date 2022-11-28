% crossValPredict performs cross validated prediction
%
% cvPredictor = crossValPredict(estimator, cv, varargin)
%
% estimator   - an fmriDataEstimator object
%
% cv    - a function handle to a method that takes an fmri_data and target
%           value as input, cv(fmri_data, Y) and returns a cvpartition
%           object. Any fields of fmri_data referenced must be preserved
%           across obj.cat and obj.get_wh_image() invocations.
%
% (optional)
%
% repartOnFit - whether cv should be reinitialized when fitting. To
%         use a predetermined cv object for fitting simply provide a
%         cvPredictor = cvPredictor.set_cvpart(cvpartition);
%         after initialization. Useful for classifier comparison.
%
% verbose  - whether to display fold information when cross validation
%
% n_paralllel - number of threads to use for parallelization
%
% crossValPredict properties:
%   cvpart  - cvpartition object used during last fit call
%   estimator
%	    - classifier used in last fit call
%   yfit    - most recent fitted values (cross validated)
%   Y       - most recent observed values
%   yfit_null - null predictions (cross validated)
%   foldEstimator
%           - estimator objects use for each fold in fit() call. Useful
%               when hyperparameters differ across folds
%
% crossValPredict methods:
%   do      - crossValPredict = cvPredictor.fit(fmri_data, Y) performs
%               cross validated predictions using fmri_data to predict Y
%   do_null - fits null model with cross validation
%   set_cvpart 
%           - sets cvpart manually (useful for reusing cv folds of other
%               training run from a different estimator)
%   repartition
%           - resets the cvpartition object.
%
%
% Notes :: 
%   crossValPredict will only work for k-fold cross validation, or at any
%   rate, cross validation schemes that have nonintersecting but complete
%   partition sets. No check is currently done to enforce this though and
%   strange errors may arrise if you try using a different partitioning
%   scheme with this function

classdef crossValPredict < crossValidator & yFit   
    properties (Dependent = true)
        cvpart;
    end
    
    properties (Hidden = true, Access = private)
        cvpart0 = [];
    end
    
    methods           
        function obj = do(obj, dat, Y)
            
            t0 = tic;
            if obj.repartOnFit || isempty(obj.cvpart)
                try
                    obj.cvpart = obj.cv(dat, Y);
                catch e
                    e = struct('message', sprintf([e.message, ...
                        '\nError occurred invoking %s.cv. Often this is due to misspecified ',...
                        'input or cvpartitioners.\nMake sure you''re using features ',...
                        'objects if you need them for passing dependence structures.'], class(obj)), 'identifier', e.identifier, 'stack', e.stack);
                    rethrow(e)
                end
            end
                        
            assert(isa(obj.cvpart, 'cvpartition'),...
                'cvpartitioner must be of type ''cvpartition'' to ensure non-overlapping partitions. Use crossValScore for performance estimation with overlapping partitions.');
            
            % do coss validation. Dif invocations for parallel vs. serial
            obj.Y = Y;
            obj.yfit = eval(sprintf('%s.empty(%d,0)',class(Y),length(Y))); % create empty array of same class as Y
            
            this_foldEstimator = cell(obj.cvpart.NumTestSets,1);
            for i = 1:obj.cvpart.NumTestSets
                this_foldEstimator{i} = copy(obj.estimator);
            end
            if obj.n_parallel <= 1            
                for i = 1:obj.cvpart.NumTestSets
                    if obj.verbose, fprintf('Fold %d/%d\n', i, obj.cvpart.NumTestSets); end

                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(dat,'image_vector')
                        train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                        test_dat = dat.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = dat(~obj.cvpart.test(i),:);
                        test_dat = dat(obj.cvpart.test(i),:);
                    end

                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    obj.yfit(obj.fold_lbls == i) = this_foldEstimator{i}.predict(test_dat, 'fast', true);
                end
            else
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool(obj.n_parallel);
                elseif pool.NumWorkers ~= obj.n_parallel
                    delete(gcp('nocreate')); 
                    parpool(obj.n_parallel);
                end
                yfit = cell(obj.cvpart.NumTestSets,1);
                parfor i = 1:obj.cvpart.NumTestSets
                    
                    train_Y = Y(~obj.cvpart.test(i));
                    if isa(dat,'image_vector')
                        train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                        test_dat = dat.get_wh_image(obj.cvpart.test(i));
                    else
                        train_dat = dat(~obj.cvpart.test(i),:);
                        test_dat = dat(obj.cvpart.test(i),:);
                    end

                    this_foldEstimator{i}.fit(train_dat, train_Y);
                    % we can always make certain assumptions about the train and test space
                    % matching which allows us to use the fast option
                    yfit{i} = this_foldEstimator{i}.predict(test_dat, 'fast', true)';
                    
                    this_foldEstimator{i} = this_foldEstimator{i}; % propogates modified handle object outside parfor loop
                    
                    if obj.verbose, fprintf('Completed fold %d/%d\n', i, obj.cvpart.NumTestSets); end
                end
                for i = 1:obj.cvpart.NumTestSets
                    obj.yfit(obj.fold_lbls == i) = yfit{i};
                end
            end
            obj.yfit = obj.yfit(:);
            
            obj.foldEstimator = this_foldEstimator;
            obj.evalTime = toc(t0);
            obj.is_done = true;
        end
        
        function obj = do_null(obj, varargin)
            if isempty(obj.cvpart)
                obj.cvpart = obj.cv(varargin{:});
                warning('cvpart not found. null predictions are not valid for yfit obtained with subsequent do() invocations');
            end
            
            if isempty(varargin)
                Y = obj.Y;
            else
                if ~isempty(obj.Y)
                    warning('obj.Y not empty, please use do_null() instead of do_null(~,Y) for best results');
                end
                
                obj.Y = varargin{2};
                Y = obj.Y;
            end
            
            obj.yfit_null = zeros(length(Y),1);
            
            for i = 1:obj.cvpart.NumTestSets                
                train_Y = Y(~obj.cvpart.test(i));
                obj.yfit_null(obj.cvpart.test(i)) = ...
                    obj.foldEstimator{i}.predict_null(obj.cvpart.TestSize(i));
            end
        end
        
        %% dependent methods
        function set.cvpart(obj, cvpart)
           obj.cvpart0 = cvpart;
           obj.yfit = [];
           obj.yfit_null = [];
           obj.evalTime = -1;
           obj.is_done = false;
           obj.repartOnFit = false;
        end
        
        function val = get.cvpart(obj)
            val = obj.cvpart0;
        end
        
        function obj = repartition(obj)
            obj.cvpart = obj.cvpart.repartition;
        end
        
        function varargout = plot(obj, varargin)
            try
                if ~isempty(varargin)
                    mkFigure = true;
                    for i = 1:length(varargin)
                        if ischar(varargin{i}) && ismember('Parent', varargin(i))
                            mkFigure = false;
                        end
                    end
                    if mkFigure, figure; end
                else
                    figure;
                end
                if isa(getBaseEstimator(obj.estimator), 'modelClf')
                    values = 1:length(obj.classLabels);

                    if ~isa(obj.Y, class(obj.yfit))
                        warning('Y is type %s but yfit is type %s. Will attempt autoconvesions.', class(obj.Y), class(obj.yfit));
                        Y = cast(obj.Y, 'like', obj.yfit);
                    else
                        Y = obj.Y;
                    end
                    cfmat = confusionmat(Y, obj.yfit);
                    
                    uniq_val = cellstr(categorical(obj.classLabels));
                    uniq_val = strrep(uniq_val,'_','');
                    
                    [uniq_val_pred, uniq_val_obs] = deal(cell(size(uniq_val)));
                    for i = 1:length(uniq_val)
                        uniq_val_pred{i} = [uniq_val{i}, ' ', sprintf('%0.3f',sum(cfmat(:,i))./sum(cfmat(:)))];
                        uniq_val_obs{i} = [uniq_val{i}, ' ', sprintf('%0.3f',sum(cfmat(i,:),2)./sum(cfmat(:)))];
                    end

                    varargout{:} = imagesc(cfmat, varargin{:});
                    set(gca,'YTickLabel',uniq_val_pred,'YTick',values,...
                        'XTickLabel', uniq_val_obs,'XTick',values, ...
                        'XTickLabelRotation', 90)
                    for i = 1:length(uniq_val)
                        for j = 1:length(uniq_val)
                            text(i,j,sprintf('%0.3f',cfmat(i,j)/sum(cfmat(:))),'HorizontalAlignment','center');
                        end
                    end
                    title('Classifier Confusion Matrix');

                elseif isa(getBaseEstimator(obj.estimator), 'modelRegressor')
                    title('Regressor Performance');
                    cla(gca); hold on;
                    b = zeros(obj.cvpart.NumTestSets,2);
                    for i = 1:obj.cvpart.NumTestSets
                        this_idx = obj.cvpart.test(i);
                        varargout{:} = plot(obj.Y(this_idx), obj.yfit(this_idx), '.', varargin{:});
                        b(i,:) = regress(obj.yfit(this_idx), [obj.Y(this_idx), ones(sum(this_idx),1)]);
                    end
                    lsline
                    b = mean(b);
                    f = @(x1) x1*b(1) + b(2);
                    h = plot(xlim, f(xlim),'-bla');
                    set(h,'LineWidth',3);
                else
                    error('Plotting is only supported for modelClf and modelRegressor type base estimators.');
                end

                xlabel('Observed')
                ylabel(sprintf('Predicted (%d-fold CV)',obj.cvpart.NumTestSets))
            
            catch e
                delete(gcf)
                rethrow(e)
            end
        end
                
    end
end
    
