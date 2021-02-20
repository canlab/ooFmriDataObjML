% crossValPredict performs cross validated prediction
%
% cvPredictor = crossValPredict(predictor, cv, varargin)
%
% predictor   - an fmriDataPredictor object
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
%   predictor
%	    - classifier used in last fit call
%   yfit    - most recent fitted values (cross validated)
%   Y       - most recent observed values
%   yfit_null - null predictions (cross validated)
%   foldPredictor
%           - predictor objects use for each fold in fit() call. Useful
%               when hyperparameters differ across folds
%
% crossValPredict methods:
%   do      - crossValPredict = cvPredictor.fit(fmri_data, Y) performs
%               cross validated predictions using fmri_data to predict Y
%   do_null - fits null model with cross validation
%   set_cvpart - sets cvpart manually (useful for reusing cv folds of other
%               training run from a different predictor/classifier)
%

classdef crossValPredict < crossValidator & yFit
    properties
        verbose = true;
    end
    
    properties (SetAccess = private)
        evalTime = -1;
        %foldPredictor = {};
    end
    
    methods
        function obj = crossValPredict(predictor, cv, varargin)
            assert(isa(predictor,'fmriDataPredictor'), 'predictor must be type fmriDataPredictor');
            
            obj.predictor = predictor;
            if ~isempty(cv), obj.cv = cv; end
            
            for i = 1:length(varargin)
                if ischar(varargin{i})
                    switch varargin{i}
                        case 'repartOnFit'
                            obj.repartOnFit = varargin{i+1};
                        case 'n_parallel'
                            obj.n_parallel = varargin{i+1};
                        case 'verbose'
                            obj.verbose = varargin{i+1};
                    end
                end
            end
        end
        
        function obj = do(obj, dat, Y)
            t0 = tic;
            if obj.repartOnFit || isempty(obj.cvpart)
                try
                    obj.cvpart = obj.cv(dat, Y);
                catch
                    keyboard
                end
            end
            
            obj.fold_lbls = zeros(length(Y),1);
            for i = 1:obj.cvpart.NumTestSets
                obj.fold_lbls(obj.cvpart.test(i)) = i;
            end
            
            obj.yfit = zeros(length(Y),1);
            this_foldPredictor = cell(obj.cvpart.NumTestSets,1);
            if obj.n_parallel == 1            
                for i = 1:obj.cvpart.NumTestSets
                    if obj.verbose, fprintf('Fold %d/%d\n', i, obj.cvpart.NumTestSets); end

                    train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                    train_Y = Y(~obj.cvpart.test(i));

                    test_dat = dat.get_wh_image(obj.cvpart.test(i));

                    this_foldPredictor{i} = obj.predictor.fit(train_dat, train_Y);
                    obj.yfit(obj.fold_lbls == i) = this_foldPredictor{i}.predict(test_dat);
                end
            else
                if ~isempty(gcp('nocreate')), delete(gcp('nocreate')); end
                parpool(obj.n_parallel);
                yfit = cell(obj.cvpart.NumTestSets,1);
                parfor i = 1:obj.cvpart.NumTestSets
                    train_dat = dat.get_wh_image(~obj.cvpart.test(i));
                    train_Y = Y(~obj.cvpart.test(i));

                    test_dat = dat.get_wh_image(obj.cvpart.test(i));

                    this_foldPredictor{i} = obj.predictor.fit(train_dat, train_Y);
                    % we can always make certain assumptions about the train and test space
                    % matching when we do cross validation. We could incorporate that here
                    % elegantly. We have to require fmriDataPredictor objects have a fast
                    % property though.
                    % this_foldPredictor{i} = this_foldPredictor{i}.fast = true;
                    yfit{i} = this_foldPredictor{i}.predict(test_dat)';
                    
                    if obj.verbose, fprintf('Completed fold %d/%d\n', i, obj.cvpart.NumTestSets); end
                end
                for i = 1:obj.cvpart.NumTestSets
                    obj.yfit(obj.fold_lbls == i) = yfit{i};
                end
            end
            
            obj.foldPredictor = this_foldPredictor;
            obj.Y = Y;
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
                obj.yfit_null(obj.cvpart.test(i)) = mean(train_Y);
            end
        end
        
        function obj = set_cvpart(obj, cvpart)
           obj.cvpart = cvpart;
           obj.yfit = [];
           obj.yfit_null = [];
           obj.evalTime = -1;
           obj.fold_lbls = [];
           obj.is_done = false;
        end
        
        function obj = repartition(obj)
            obj.cvpart = obj.cvpart.repartition;
        end
        
        function obj = compress(obj)
            for i = 1:length(obj.foldPredictor)
                obj.foldPredictor{i} = obj.foldPredictor{i}.compress();
            end
        end
    end
end
    
