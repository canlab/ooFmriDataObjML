close all; clear all;

%% import libraries and their dependencies

addpath('/dartfs-hpc/rc/home/m/f0042vm/software/spm12'); % canlabCore dep

addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/CanlabCore')); % canlab_single_trails* and ooFmriDataObjML dep
addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/Neuroimaging_Pattern_Masks')); % canlab_single_trails* and ooFmriDataObjML dep
addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/MasksPrivate')); % canlab_single_trails* and ooFmriDataObjML dep

addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/CanlabPrivate')); % canlab_single_trails* and ooFmriDataObjML dep

addpath(genpath('/dartfs/rc/lab/C/CANlab/labdata/projects/canlab_single_trials_for_git_repo')); % canlab_single_trials dep
addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/canlab_single_trials')); % data repo
addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/canlab_single_trials_private')); % data repo

addpath('/dartfs-hpc/rc/home/m/f0042vm/software/combat/ComBatHarmonization/Matlab/scripts'); % ooFmriDataObjML dep
addpath(genpath('/dartfs-hpc/rc/home/m/f0042vm/software/canlab/ooFmriDataObjML')); % an MVPA modeling framework

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

%% load dataset
%
% This dataset contains subject level task contrasts for 18 different tasks
% across 9 different studies (approximately, some conditions only exist in 
% one study, but in those cases there's another study with a somewhat 
% similar condition). This is the dataset we'll use for multiclass
% classification.
%
% Note that this will download a file called 
% kragel_2018_nat_neurosci_270_subjects_test_images.mat in your current
% working directory. If you already have it you can try this instead
% imgs = importdata(which('kragel_2018_nat_neurosci_270_subjects_test_images.mat'))
%
imgs = load_image_set('kragel18_alldata');

disp(unique(imgs.metadata_table.Domain,'stable'))
disp(unique(imgs.metadata_table.Subdomain,'stable'))

%% Build a domain classifier
% much of what follows is similar to the estimateBestRegionPerformance demo
% in how it sets up bayesian optimization and cvpartition objects

classes = unique(imgs.metadata_table.Subdomain);
nclasses = length(classes);

alg = multiclassLinearSvmClf('NClasses', nclasses, 'regularization', 'ridge');

% list hyperparameters
disp(alg.get_params())

inner_cv = @(X,Y) cvpartition(X.metadata.Studynumber, 'KFold', 5); % we want to balance partitions across studies
% notice that inner_cv is a function, not a cvpartition2 object. Calling
% inner_cv on some data will return a appropriately constructed
% cvpartition2 object. This is important because crossValidator objects
% need instructions on how to generate these things, not specific instances
% of them.

% next two lines are basically the same as if you were invoking the
% bayesopt native matlab function. The hyperparameters we're optimizing
% were selected from alg.get_params() output and correspond to the
% different ecoc classifiers. For theoretical background consider referring
% to mathworks documentation here: https://www.mathworks.com/help/stats/classificationecoc.html#bue4w15
% Alternatively the scikit-learn documentation may also be helpful, see
% here: https://scikit-learn.org/stable/modules/multiclass.html#ovo-classification
lambda = optimizableVariable('lambda',10.^[-3,4], 'Type', 'real', 'Transform', 'log');
bayesOptOpts = {lambda, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
     'MaxObjectiveEvaluations', 15, 'UseParallel', true};

% let's start up a parallel pool which controls how many parallel threads
% bayesOpt will use
parpool(4);

bo_alg = bayesOptCV(alg, inner_cv, @get_hinge_loss, bayesOptOpts);

% test algorithm
% notice that I cast imgs.metadata_table.Domain to a categorical variable
dat = features(imgs.dat', table(imgs.metadata_table.Studynumber, 'VariableNames',{'Studynumber'})); % this is an "extended double" that is just a double with metadata in the dat.metadata field
bo_alg.fit(dat, categorical(imgs.metadata_table.Subdomain)); % note handle invocation doesn't use assignment operator (i.e. there's no '=' sign).

%% configure outer CV loop and data preprocessing
% here we'll define some masks and apply an L2 norm transformation to make
% this a pattern classifier that's agnostic w.r.t. overall image intensity.
% We use L2 rather than z-score because it doesn't alter representaitonal
% geometry.

% this is a transformer that applys a gray matter mask as a first step
gray_mask = fmri_mask_image('gray_matter_mask.img');
funhan = @(x1)apply_mask(x1, gray_mask);
mask2GrayMat = functionTransformer(funhan);

% this uses the generic functionTransformer to apply an anonymous function
% to our data. The function L2-norms images.
funhan = @(x1)rescale(x1, 'l2norm_images');
l2 = functionTransformer(funhan);

% mask2Region and L2norm may take fmri_data objects as input, but our bayes 
% optimized multiclassLinearSvmClf does not, so we also need a transformer 
% that takes fmri_data objects as inputs and returns a features object.
% This object saves a bunch of metadata on fmri_data objects in its
% brainModel property, which is useful if you want to project your patterns
% back into brain space later.
% note how the metadataconstructor_funhan defines what metadata gets
% packaged into the features.metadata field. The invocation here is
% trivial, but when you have multiple items you need in your features
% metadata (e.g. subject_ids and study_ids), it can be helpful to insert a 
% table constructor object in there instead so that your data is labeled.
% Notice that this is just an anonymous function that does what we did when
% we invoked features() above when tsting bo_alg.
fmriDat2Feat = fmri2VxlFeatTransformer('metadataConstructor_funhan', @(X) table(X.metadata_table.Studynumber, 'VariableNames', {'Studynumber'}));

% the next line creates a meta algorithm that combines mask2Region and
% bayes optimized PLS into a single pipeline. The syntax is pretty similar
% to scikit learn's here, although I think scikit-learn might not 
% interleave names and elements but, rather sort them sequentially instead.
bo_alg_full = pipeline({{'mask', mask2GrayMat}, {'l2norm', l2}, {'fmriDat2Feat', fmriDat2Feat}, {'bayesOptPLS', bo_alg}});

% we don't need to run this here, but this is a helpful test that the code 
% thus far works as intended. This is also the function who's performance 
% we want to ultimately estimate, so we'd need to fit it later to test on
% bmrk3pain anyway.
bo_alg_full.fit(imgs, categorical(imgs.metadata_table.Subdomain))

% check which lambda was best
% all will be the same, so we just take the first.
fprintf('best region: %s\n', bo_alg_full.getBaseEstimator.lambda{1});

%% Evaluate overall model performance
%
% notice that we don't use hinge loss for evaluating the performance. There
% are various suitable metrics but hinge_loss isn't interpretable at all.
% Check wikipedia for f1_macro interpretation.
outer_cv = @(X,Y) cvpartition(X.metadata_table.Studynumber, 'KFold', 5); 
cv = crossValScore(bo_alg_full, outer_cv, @get_f1_macro, 'verbose', true);

cv.do(imgs, categorical(imgs.metadata_table.Subdomain))
cv.do_null(); % this tests the null performance given our partitioning scheme
cv.plot(); % this will only work if outer_cv partitions are non-overlapping.

%% plot brain model

brainModel = bo_alg_full.transformers{end}.brainModel;
brainModel.dat = bo_alg_full.getBaseEstimator.B;
for i = 1:size(brainModel.dat,2)
    f = figure;    
    brainModel.get_wh_image(i).montage()
    sgtitle(bo_alg_full.getBaseEstimator.classLabels(i));
end