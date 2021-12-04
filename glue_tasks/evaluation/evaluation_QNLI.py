from fairseq.models.roberta import RobertaModel

"""
Evaluation for QNLI task. The metric it the Accuracy.
"""
roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/roberta_results/checkpoints/',
    checkpoint_file='checkpoint_best_qnli_large.pt',
    data_name_or_path='../glue_tasks/QNLI-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('../glue_tasks/glue_data/QNLI/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(target, prediction_label, sent1, sent2)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))

