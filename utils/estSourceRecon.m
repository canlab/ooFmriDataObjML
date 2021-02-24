% function src = estSourceRecon(mdlObj, target_fmri_data)
%
% Performs source reconstruction using a fitted mdlObj of either type
% crossValidator or fmriDataEstimator. If type crossValidator then the
% source reconstruction will be cross validated, and it's assumed
% target_fmri_data is the training data.
function src = estSourceRecon(mdlObj, target_fmri_data)
    % valid predictors must have uncorrelated sources. For details see 
    % Haufe S, Meinecke F, Görgen K, Dähne S, Haynes JD, Blankertz B, 
    % Bießmann F. (2014) "On the interpretation of weight vectors of linear
    % models in multivariate neuroimaging". Neuroimage 87. 
    valid_predictors = {'pls','pcr','mlpcr'};
    
    % predictors with cv_* prefixed are borrowed from the canlabCore
    % predict() function, but should otherwise be the same.
    valid_predictors = [valid_predictors, cellfun(@(x1)(['cv_', x1]), valid_predictors,'UniformOutput',false)];

    basePredictor = get_base_predictor(mdlObj);
    assert(ismember(basePredictor.algorithm_name, valid_predictors), ...
        sprintf('%s is not a supported predictor', basePredictor.algorithm_name));
    
    if isa(mdlObj, 'crossValidator')
        assert(mdlObj.is_done, 'Please run mdlObj.do(X,Y) before estimating source reconstruction.');
        
        assert(length(mdlObj.cvpart.training(1)) == size(target_fmri_data.dat,2),...
            'Target_fmri_data does not match training data. Cannot continue with cross validated source reconstruction. Try supplying an fmriDataPredictor as the mdlObj.');
        % compute cov(target_fmri_data.dat, yfit) in a cross validated
        % manner, but do it on transformed target_fmri_data, not raw
        % target_fmri_data if any transformers are used by the predictor.
        src = cell(mdlObj.cvpart.NumTestSets,1);
        parfor k = 1:mdlObj.cvpart.NumTestSets
            this_target = target_fmri_data.get_wh_image(mdlObj.cvpart.test(k));
            
            predictor = mdlObj.foldPredictor{k};
            yfit = predictor.predict(this_target);
            
            if isa(predictor, 'pipeline')
                this_target = predictor.transform(this_target);
            end
            
            basePredictor = get_base_predictor(predictor);
            this_target = this_target.apply_mask(basePredictor.weights);
            this_src = this_target.get_wh_image(1);

            n = size(this_target.dat,2);
            this_target.dat = this_target.dat - mean(this_target.dat,2);
            yfit = yfit - mean(yfit);
            
            this_src.dat = 1/(n-1)*this_target.dat*yfit(:);
            
            src{k} = this_src;
        end
        src = cat(src{:});
    elseif isa(mdlObj, 'fmriDataPredictor')
        % compute cov(target_fmri_data.dat, yfit) in a non cross validated
        % manner (e.g. for target data that is different from training
        % data), but do it on transformed target_fmri_data, not raw
        % target_fmri_data if any transformers are used by the predictor.
        
        if isa(mdlObj, 'pipeline')
            target_fmri_data = mdlObj.transform(target_fmri_data);
            yfit = mdlObj.predictor.predict(target_fmri_data);
        else
            yfit = mdlObj.predict(target_fmri_data);
        end
        
        basePredictor = get_base_predictor(mdlObj);
        target_fmri_data = target_fmri_data.apply_mask(basePredictor.weights);
        src = target_fmri_data.get_wh_image(1);
        
        n = size(target_fmri_data.dat,2);
        target_fmri_data.dat = target_fmri_data.dat - mean(target_fmri_data.dat,2);
        yfit = yfit - mean(yfit);
        
        src.dat = 1/(n-1)*target_fmri_data.dat*yfit(:);
    else
        error('mdlObj must be type crossValidator or fmriDataPredictor');
    end
end

function basePred = get_base_predictor(predictor)
    assert(isa(predictor,'fmriDataPredictor') || isa(predictor,'crossValidator'),'Predictor is not an fmriDataPredictor');
    basePred = predictor;
    
    fnames = fieldnames(predictor);
    pfname = ismember(fnames,'predictor');
    if any(pfname)
        basePred = get_base_predictor(basePred.(fnames{pfname}));
    end
end

