import torch
from fairseq.models.roberta import RobertaModel

"""
Evaluation for CoLA task. The metric it the Accuracy.
Please notice that you can get the pretrained large mnli model from torch hub:
in this case, remember to change the head name to 'mnli'
"""

# Large MNLI model from torch hub
# roberta = torch.hub.load("pytorch/fairseq:main", "roberta.large.mnli")
# roberta.eval()

roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/ml/fairseq/checkpoints/',
    checkpoint_file='checkpoint_best_mnli_base.pt',
    data_name_or_path='../MNLI-bin'
)
roberta.cuda()
roberta.eval()

# 0 = contradiction 1 = neutral 2 = entailment
# test both on matched and mismatched test sets.
ncorrect, nsamples = 0, 0
with torch.no_grad():
    with open('../glue_data/MNLI/dev_mismatched.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
            if target == "contradiction":
                target_value = 0
            elif target == "neutral":
                target_value = 1
            elif target == "entailment":
                target_value = 2
            # if base 'sentence_classification_head' else if large 'mnli'
            prediction_label = roberta.predict('sentence_classification_head', roberta.encode(sent1, sent2)).argmax().item()
            print(target_value, prediction_label, sent1, sent2, target)
            ncorrect += int(prediction_label == target_value)
            nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))


