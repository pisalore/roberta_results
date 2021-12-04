from fairseq.models.roberta import RobertaModel
"""
Evaluation for SST task. The metric it the Accuracy.
"""
roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/roberta_results/checkpoints/',
    checkpoint_file='checkpoint_best_sst2_large.pt',
    data_name_or_path='../glue_tasks/SST-2-bin'
)
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('../glue_tasks/glue_data/SST-2/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent, target = tokens[0], tokens[1]
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(prediction_label, target, sent)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))