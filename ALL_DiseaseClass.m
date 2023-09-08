model_dir = '/home/changal/Documents/bmi_allen_rishi/Model_classification/disease_prediction';
coh_dir = fullfile(model_dir,'cohbin');
psd_dir = fullfile(model_dir,'psdbin');

models = {'LR','LDA','DT','RF','NB','KNN','Train','Me','Global','Uni','Hard'};
%% Predict with psd
model_csv = load(fullfile(psd_dir,'model_accuracies.csv'));

figure
nexttile
mean_bar = mean(model_csv,1);
bar(mean_bar);
err = std(model_csv,1)./sqrt(size(model_csv,1));
hold on
errorbar(1:11,mean_bar,err,'k','LineStyle','none')
ylim([0 1])
ylabel('Accuracy')
xticklabels(models)
title('PSD - Healthy vs Stroke')

%% Predict with coh
model_csv = load(fullfile(coh_dir,'model_accuracies.csv'));

nexttile
mean_bar = mean(model_csv,1);
bar(mean_bar);
err = std(model_csv,1)./sqrt(size(model_csv,1));
hold on
errorbar(1:11,mean_bar,err,'k','LineStyle','none')
ylim([0 1])
ylabel('Accuracy')
xticklabels(models)
title('Coh - Healthy vs Stroke')
sgtitle('ALL phases, All stim states, All freq Bands, Electrode')
