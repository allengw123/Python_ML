pre_dir = '/home/changal/Documents/bmi_allen_rishi/Model_classification/disease_prediction';
post_dir = '/home/changal/Documents/bmi_allen_rishi/Model_prediction/disease_prediction';

stim_states = {'Stim', 'Sham'};
timepoints = {'intra5','intra15','post'};
models = {'LR','LDA','DT','RF','NB','KNN','ADA','Train','Me','Global','Uni','Hard'};

%%
sham_model_csv = load(fullfile(pre_dir,'Sham','disease_prediction','psdbin','model_accuracies.csv'));
stim_model_csv = load(fullfile(pre_dir,'Stim','disease_prediction','psdbin','model_accuracies.csv'));


sham_model_mean = mean(sham_model_csv,1);
stim_model_mean = mean(stim_model_csv,1);

figure
nexttile
hdl = plot([sham_model_mean' stim_model_mean']','-o');
err = [std(sham_model_csv,1)./sqrt(size(sham_model_csv,1))' ;std(stim_model_csv,1)./sqrt(size(stim_model_csv,1))'];
hold on
errorbar(1:2,[sham_model_mean' stim_model_mean']',err,'k','LineStyle','none')
ylim([0 1])
ylabel('Accuracy')
xlim([0 3])
xticks(1:2)
xticklabels({'Sham','Stim'})
title('Pre')
legend(hdl,models)

for i = 1:numel(timepoints)
    sham_model_csv = readtable(fullfile(post_dir,['Sham_',timepoints{i}],'results.csv'));
    stim_model_csv = readtable(fullfile(post_dir,['Stim_',timepoints{i}],'results.csv'));
    
    for m = 1:numel()
    sham_model_mean = mean(sham_model_csv,1);
    stim_model_mean = mean(stim_model_csv,1);
    
    nexttile
    hdl = plot([sham_model_mean' stim_model_mean']','o');
    err = [std(sham_model_csv,1)./sqrt(size(sham_model_csv,1))' ;std(stim_model_csv,1)./sqrt(size(stim_model_csv,1))'];
    hold on
    errorbar(1:2,[sham_model_mean' stim_model_mean']',err,'k','LineStyle','none')
    ylim([0 1])
    ylabel('Accuracy')
    xlim([0 3])
    xticks(1:2)
    xticklabels({'Sham','Stim'})
    title('Pre')
    legend(hdl,models)