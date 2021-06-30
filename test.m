% this script tests some basic functionality for a number of methods in
% ooFmriDataPredictML repo using the nsf dataset from the
% canlab_single_trials repo
%
% could be used as the starting point for a unit testing framework, but
% it's not quite that comprehensive at this point.

close all; clear all;

%% import libraries and their dependencies

addpath('/projects/bope9760/spm12'); % canlabCore dep

addpath(genpath('/projects/bope9760/software/canlab/CanlabCore')); % canlab_single_trails* and ooFmriDataObjML dep
addpath(genpath('/projects/bope9760/software/canlab/Neuroimaging_pattern_masks')); % canlab_single_trails* and ooFmriDataObjML dep
addpath(genpath('/projects/bope9760/software/canlab/MasksPrivate')); % canlab_single_trails* and ooFmriDataObjML dep

addpath(genpath('/projects/bope9760/software/canlab/CanlabPrivate')); % canlab_single_trails* and ooFmriDataObjML dep

addpath(genpath('/work/ics/data/projects/wagerlab/labdata/projects/canlab_single_trials_for_git_repo/')); % canlab_single_trials dep
addpath(genpath('/projects/bope9760/software/canlab/canlab_single_trials')); % data repo
addpath(genpath('/projects/bope9760/software/canlab/canlab_single_trials_private')); % data repo

addpath('/projects/bope9760/software/combat/ComBatHarmonization/Matlab/scripts'); % ooFmriDataObjML dep
addpath(genpath('/projects/bope9760/software/canlab/ooFmriDataObjML')); % an MVPA modeling framework

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end
parpool(8)

%% load and simplify some data to work with

nsf = load_image_set('nsf');
nsfStim = avgByStimLvl2(nsf, double(categorical(nsf.metadata_table.subject_id)), nsf.metadata_table.T);

ns = fmri_mask_image('/projects/bope9760/combat/resources/neurosynth_pain_map/association-test_z_FDR_0.05.nii');
nsf = nsfStim.apply_mask(ns);

%% cvpartitioner2: Test
id =  sort(kron(1:10,ones(1,3)));
partitioner = cvpartition2([ones(length(id)/2,1); 2*ones(length(id)/2,1)], 'GroupKFold', 5, 'Group', id, 'Stratify', true);
partitioner
disp('Training sets');
for i = 1:5
    disp(partitioner.training(i)');
end
disp('Test sets');
for i = 1:5
    disp(partitioner.test(i)');
end

%% cvpartitionShuffleSplit: Test
partitioner = cvpartitionShuffleSplit([ones(length(id)/2,1); 2*ones(length(id)/2,1)], ...
    'Splits', 5, 'HoldOut', 0.8, 'Stratify', true, 'Group', id);
partitioner
disp('Training sets');
for i = 1:5
    disp(partitioner.training(i)');
end
disp('Test sets');
for i = 1:5
    disp(partitioner.test(i)');
end

partitioner = partitioner.repartition();

%% crossValPredict: Test 1, serial
alg = plsRegressor('numcomponents', 10);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValPredict(alg, outer_cvpartitioner, @get_mse, 'verbose', true);

X = features(nsf.dat', nsf.metadata_table.subject_id);
cv.do(X, nsf.metadata_table.T)
cv.do_null();
cv.plot()

%% crossValPredict: Test 2, parallel
alg = plsRegressor('numcomponents', 10);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValPredict(alg, outer_cvpartitioner, @get_mse, 'verbose', true, 'n_parallel', 2);

X = features(nsf.dat', nsf.metadata_table.subject_id);
cv.do(X, nsf.metadata_table.T)
cv.do_null();
cv.plot()

%% crossValPredict to crossValScore conversion
crossValScore(cv, @get_mse).do_null().plot()

%% crossValScore: Test 1, serial
alg = plsRegressor('numcomponents', 10);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValScore(alg, outer_cvpartitioner, @get_mse, 'verbose', true);

X = features(nsf.dat', nsf.metadata_table.subject_id);
cv.do(X, nsf.metadata_table.T)
cv.do_null();
cv.plot();

%% crossValScore: Test 2, parallelization
alg = pcrRegressor('numcomponents', Inf);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValScore(alg, outer_cvpartitioner, @get_mse, 'verbose', true, 'n_parallel', 2);

X = features(nsf.dat', nsf.metadata_table.subject_id);
cv.do(X, nsf.metadata_table.T)
cv.do_null();

fprintf('PCR without optimization, MSE = %0.3f, R2 = %0.3f\n', mean(cv.scores), 1 - mean(cv.scores)/mean(cv.scores_null));

%% crossValScore: Test 3, overlapping partitions
alg = plsRegressor('numcomponents', 10);
outer_cvpartitioner = @(X,Y) cvpartitionShuffleSplit(length(Y), 'Splits', 5, 'HoldOut', 0.2, 'Group', X.metadata);
cv = crossValScore(alg, outer_cvpartitioner, @get_mse, 'verbose', true);

X = features(nsf.dat', nsf.metadata_table.subject_id);
cv.do(X, nsf.metadata_table.T)
cv.do_null();

%% crossValScore to crossValPredict conversion
crossValPredict(cv).plot()

%% test bayesOptCV
% turn off warnings related to untested code
warning('off', 'cvpartitionMemoryImpl2:updateParams');

alg = pcrRegressor();
inner_cvpartitioner = @(X,Y) cvpartitionShuffleSplit(length(Y), 'Splits', 5, 'HoldOut', 0.2, 'Group', X.metadata);
    
dims = optimizableVariable('numcomponents',[1,80], 'Type', 'integer');
bayesOptOpts = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
     'MaxObjectiveEvaluations', 30, 'UseParallel' 0, 'verbose', 0};
bo_alg = bayesOptCV(alg, inner_cvpartitioner, @get_mse, bayesOptOpts);

bo_alg.fit(X, nsf.Y);

%% test nested CV and parallelized bayesOptCV
bayesOptOpts = {dims, 'AcquisitionFunctionName', 'expected-improvement-plus', ...
     'MaxObjectiveEvaluations', 15, 'UseParallel' true, 'verbose', 0, 'PlotFcn', {}};
bo_alg = bayesOptCV(alg, inner_cvpartitioner, @get_mse, bayesOptOpts);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValScore(bo_alg, outer_cvpartitioner, @get_mse, 'verbose', true);

cv.do(X, nsf.metadata_table.T)
cv.do_null();

fprintf('PCR with bayesian optimization, MSE = %0.3f, R2 = %0.3f\n', mean(cv.scores), 1 - mean(cv.scores)/mean(cv.scores_null));
warning('on', 'cvpartitionMemoryImpl2:updateParams');


%% test gridSearchCV
% turn off warnings related to untested code
warning('off', 'cvpartitionMemoryImpl2:updateParams');

alg = pcrRegressor();
inner_cvpartitioner = @(X,Y) cvpartitionShuffleSplit(length(Y), 'Splits', 5, 'HoldOut', 0.2, 'Group', X.metadata);
    
gridPoints = table((1:30)','VariableNames',{'numcomponents'});
gs_alg = gridSearchCV(alg, gridPoints, inner_cvpartitioner, @get_mse, 'verbose', true);

gs_alg.fit(X, nsf.Y);
warning('on', 'cvpartitionMemoryImpl2:updateParams');


%% test nested CV and parallelized gridSearchCV
gridPoints = table(sort(randperm(80))','VariableNames',{'numcomponents'});
gs_alg = gridSearchCV(alg, gridPoints, inner_cvpartitioner, @get_mse, 'n_parallel', 8);
outer_cvpartitioner = @(X, Y)cvpartition2(ones(length(Y),1), 'GroupKFold', 5, 'Group', X.metadata);
cv = crossValScore(gs_alg, outer_cvpartitioner, @get_mse, 'verbose', true);

cv.do(X, nsf.metadata_table.T)
cv.do_null();

fprintf('PCR with grid search optimization, MSE = %0.3f, R2 = %0.3f\n', mean(cv.scores), 1 - mean(cv.scores)/mean(cv.scores_null));
warning('on', 'cvpartitionMemoryImpl2:updateParams');