from fairseq.models.roberta import RobertaModel

"""
Evaluation for MRPC task. The metric it the Accuracy and F1 score.
"""
roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/roberta_results/checkpoints/',
    checkpoint_file='checkpoint_best_mrpc_base.pt',
    data_name_or_path='../glue_tasks/MRPC-bin'
)
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples, pred_positives, true_positives, false_negatives = 0, 0, 0, 0, 0
roberta.cuda()
roberta.eval()
with open('../glue_tasks/glue_data/MRPC/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[3], tokens[4], tokens[0]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(target, prediction_label, sent1, sent2)
        if int(prediction_label) == 1:
            pred_positives += 1
        if prediction_label == target and int(target) == 1:
            true_positives += 1
        if int(target) == 1 and prediction_label != target:
            false_negatives += 1
        ncorrect += int(prediction_label == target)
        nsamples += 1
precision = float(true_positives)/float(pred_positives)
recall = float(true_positives)/float(true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print('| Accuracy: ', float(ncorrect)/float(nsamples))
print('| Precision: ', precision)
print('| Recall: ', recall)
print('| F1 Score: ', f1_score)

