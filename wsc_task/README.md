### WSC task

For WNLI, Facebook reasearcher used the original "WSC" data in [SuperGLUE](https://super.gluebenchmark.com/). SuperGLUE benchmark is styled after GLUE, with more challenging tasks.

**WSC (Winograd Schema Challenge, Levesque et al., 2012)** is a coreference resolution task in
which examples consist of a sentence with a pronoun and a list of noun phrases from the sentence.
The system must determine the correct referrent of the pronoun from among the provided choices.

1. **Download the dataset**
    ```bash
    wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip
    unzip WSC.zip
    rm WSC.zip
    ```

2. **Copy RoBERTa dictionary in the same directory**
    ```bash
    wget -O WSC/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
    ```

3. **Fine tune on WSC**
    ```bash
    ./roberta_wsc_finetuning.sh
    ```
    It will save your model under `wsc_task/checkpoints/checkpoint_best.pt`

4. **Evaluate**
    ```bash
    python roberta_wsc_evaluation.py
    ```