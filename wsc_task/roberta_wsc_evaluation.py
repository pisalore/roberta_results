from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils 
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
roberta.cuda()
nsamples, ncorrect = 0, 0
for sentence, label in wsc_utils.jsonl_iterator('WSC/val.jsonl', eval=True):
    pred = roberta.disambiguate_pronoun(sentence)
    print(pred, label, sentence)
    nsamples += 1
    if pred == label:
        ncorrect += 1
print('Accuracy: ' + str(ncorrect / float(nsamples)))
