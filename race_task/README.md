### RACE task
Reproduce the RoBERTA results on [RACE dataset](https://www.cs.cmu.edu/~glai1/data/race/)

1. **Download the [dataset](https://www.cs.cmu.edu/~glai1/data/race/).** \
You have to compile a google form and wait for an email with the link.

2. **Extract the dataset**
    ```shell
    tar -xvf RACE.tar.gz
    rm RACE.tar.gz
    python ./../fairseq/examples/roberta/preprocess_RACE.py --input-dir RACE-extracted
    ```
3. **Preprocess dataset**
    ```shell
    ./../fairseq/examples/roberta/preprocess_RACE.sh RACE-extracted/ RACE-bin/
    ```

3. **Finetune**\
    Models will be saved in `checkpoints/` directory.
    ```shell
    ./roberta_race_finetuning.sh 
    ```

3. **Evaluate and getr results**
    ```shell
    ./roberta_race_evaluate.sh 
    ```


