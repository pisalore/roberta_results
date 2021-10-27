### GLUE tasks

Reproduce the glue tasks results with RoBERTa. See the [RoBERTa submission at GLUE](https://gluebenchmark.com/submission/JuLiHrAkS9VSQRh1W6TJ9V9SOu23/-Lk5ZrckAabWVeQBoxrA).\
In this doc I will synthetize steps and their motivations.

"The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems."

In fact, GLUE consists of 9 different dataset, each with different pourposes and origins. Please check the [official GLUE paper](https://openreview.net/pdf?id=rJ4km2R5t7) for further details.


1. Download all tasks dataset.
    ```shell
    python download_glue_data.py --data_dir glue_data --tasks all
    ```
2. Manually split `MRPC` data.
    You will notice that downloading `glue_data` you will be advised that it was impossible to download the dev set for `MRPC`. So, do the following:

    - Convert and rename `msr_paraphrase_train.txt` to `train.tsv`.
    - Rename `test.tsv` to `dev.tsv`

    These operations are necessary for data preprocessing.

3. Preprocess data\
    As you can read in RoBERTa paper, data are preprocessed using **Byte-Pair Encoding (BPE)** for a better data handling.
    ```shell
    ./../fairseq/examples/roberta/preprocess_GLUE_tasks.sh glue_data ALL
    ```
    You should obtain 8 different `*-bin` folders, each one for a GLUE task data (WNLI is not present). These folders contain the effective dataset which will be used during fine-tuning.

4. Pre-trained models finetuning for each task\
    Now we can finetune our models on datasets following the hyperparameters described in RoBERTa paper. I provided different shell scripts for each task, however you have two possibilities:
    - Always use the shell scripts, changing the architecture accordigly. For example, if you want to finetune using `roberta.large`:
    ```shell
    ...
    ROBERTA_PATH="/home/lpisaneschi/ml/fairseq/roberta.large/model.pt"
    ...
    --arch roberta_large
    ```
    - Run this command, working only if you want to finetune `roberta.base`.
    ```shell
    CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train --config-dir ../fairseq/examples/roberta/config/finetuning --config-name <task-name> task.data=/path/to/roberta_results/glue_tasks/MRPC-bin checkpoint.restore_file=/path/to/roberta.base/model.pt
    ```
    The main difference is that, with the shell script, you can finetune with the architecture you prefer, and that all `checkpoint_*.pt` files generated will be saved in the `SAVE_DIR` you provide; on the other hand,  with the second option you can train only `roberta.base` and only `checkpoint_last.pt` and `checkpoint_best.pt` will be saved, in a `checkpoints/` directory.


    Each script is set with the hyperparameters described in the official papers. Any possible difference is documented in the script itself.

5. Evaluation\
    In order to evaluate the obtained finetuned models, you find different python scripts in this repo in `glue_tasks/inference/` directory.
    You have simply to load the correct finetuned model for you task 'checkpoint_best.pt`:
    ```python
    roberta = RobertaModel.from_pretrained(
        '/path/to/checkpoints/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='../task-bin'
    )
    ```

    Note that for `MNLI` inference you can load `roberta.large.mnli` model also from  `toch.hub`:
    ```python
    # Large MNLI model from torch hub
    roberta = torch.hub.load("pytorch/fairseq:main", "roberta.large.mnli")
    roberta.eval()

    # Your model
    robe'/path/to/checkpoints/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='../MNLI-bin'
    )

