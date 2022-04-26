# Generating Extended and Multilingual Summaries with Pre-trained Transformers

Code for the paper Generating Extended and Multilingual Summaries with Pre-trained Transformers accepted at LREC 2022.

## Getting started

Create the environnement, activate and install requirements.

```bash
conda create -n mdmls python=3.7
conda activate mdmls
pip install -r requirements.txt
```

## WikinewsSum dataset

Please refer to https://github.com/airklizz/wikinewssum to download the dataset.

Place the `train.json`, `validation.json`, and `test.json` files in the `wikinewssum/` folder.

## Preprocessing

Prepare the dataset to fine-tune an abstractive model using an extractive pre-abstractive step.

```bash
python mdmls/main.py preprocess extractive-bert \
    wikinewssum/train.json \
    wikinewssum/train_pre_abstractive.json \
    --model-checkpoint distilbert-base-multilingual-cased \
    --pre-abstractive \
    --abstractive-model-checkpoint google/mt5-small
```

Tokenize the dataset.

```bash
python mdmls/main.py preprocess tokenize \
    wikinewssum/train_pre_abstractive.json \
    wikinewssum/train_pre_abstractive_tokenized.json \
    --source distilbert-base-multilingual-cased_extractive_summary \
    --model-checkpoint google/mt5-small
```

> The same steps need to be performed for the validation set.

## Fine-tuning

Use the command line interface to fine-tune a new model of the WikinewsSum dataset.

```bash
python mdmls/main.py train run \
    --train-data-files wikinewssum/train_pre_abstractive_tokenized.json \
    --validation-data-files wikinewssum/validation_pre_abstractive_tokenized.json \
    --training-scenario "new-fine-tuning" \
    --model-checkpoint google/mt5-base
```

To see all the parameters.

```bash
> python mdmls/main.py train run --help
Usage: main.py train run [OPTIONS]

Options:
  --train-data-files TEXT
  --validation-data-files TEXT
  --training-scenario TEXT
  --model-checkpoint TEXT         [default: google/mt5-small]
  --batch-size INTEGER            [default: 8]
  --gradient-accumulation-steps INTEGER
                                  [default: 1]
  --num-train-epochs INTEGER      [default: 8]
  --learning-rate FLOAT           [default: 5.6e-05]
  --weight-decay FLOAT            [default: 0.01]
  --save-total-limit INTEGER      [default: 3]
  --push-to-hub / --no-push-to-hub
                                  [default: push-to-hub]
  --language TEXT
  --max-number-training-sample INTEGER
  --help                          Show this message and exit.
```

| option                       | description                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| --language                   | if specified, only the samples of the specified language are kept. For example: `--language en` to train on the English samples only |
| --max-number-training-sample | if specified, limit the number of training sample to the value                                                                       |

## Evaluation

### ROUGE scores

| Methods                                                                | Metrics | English | German | French | Spanish | Portuguese | Polish | Italian | All Languages |
| ---------------------------------------------------------------------- | ------- | ------- | ------ | ------ | ------- | ---------- | ------ | ------- | ------------- |
| Extractive Summarisation                                               |
| DistilmBERT                                                            | R-1 F   | 41.37   | 29.37  | 29.80  | 29.70   | 29.62      | 24.83  | 35.18   | 33.51         |
|                                                                        | R-2 F   | 14.35   | 8.42   | 12.57  | 12.52   | 14.33      | 10.48  | 12.59   | 12.34         |
|                                                                        | R-L F   | 19.66   | 13.65  | 17.10  | 17.07   | 18.75      | 15.03  | 18.43   | 17.30         |
| mBERT                                                                  | R-1 F   | 41.37   | 29.74  | 29.74  | 35.50   | 29.66      | 24.82  | 34.93   | 33.60         |
|                                                                        | R-2 F   | 14.48   | 8.70   | 12.62  | 13.31   | 14.51      | 10.55  | 12.68   | 12.51         |
|                                                                        | R-L F   | 19.63   | 13.83  | 17.13  | 18.10   | 18.86      | 15.07  | 18.86   | 17.36         |
| XLM-RoBERTa                                                            | R-1 F   | 40.92   | 29.00  | 29.70  | 35.40   | 29.39      | 24.74  | 35.68   | 33.27         |
|                                                                        | R-2 F   | 14.22   | 8.33   | 12.52  | 13.03   | 14.13      | 10.49  | 12.54   | 12.26         |
|                                                                        | R-L F   | 19.66   | 13.54  | 17.07  | 18.05   | 18.43      | 15.03  | 19.54   | 17.26         |
| Oracle                                                                 | R-1 F   | 49.50   | 37.21  | 34.41  | 42.24   | 35.32      | 29.89  | 41.85   | 40.29         |
|                                                                        | R-2 F   | 25.72   | 15.77  | 17.31  | 20.89   | 21.40      | 15.72  | 19.94   | 20.35         |
|                                                                        | R-L F   | 22.67   | 15.93  | 17.38  | 20.54   | 19.19      | 15.33  | 18.61   | 19.16         |
| Abstractive Summarisation after Oracle Pre-Abstractive Extractive Step |
| mT5 Cross-lingual zero-shot transfer                                   | R-1 F   | 44.26   | 9.13   | 9.63   | 11.23   | 10.77      | 6.93   | 9.71    | 19.99         |
|                                                                        | R-2 F   | 21.73   | 2.85   | 2.52   | 3.71    | 3.26       | 1.76   | 2.48    | 8.53          |
|                                                                        | R-L F   | 24.25   | 6.31   | 6.32   | 7.81    | 7.51       | 5.05   | 6.53    | 11.92         |
| mT5 In-language multi-task                                             | R-1 F   | 43.19   | 33.14  | 36.92  | 37.69   | 34.54      | 27.95  | 37.00   | 37.05         |
|                                                                        | R-2 F   | 21.33   | 13.47  | 17.40  | 17.46   | 18.05      | 13.65  | 13.87   | 17.51         |
|                                                                        | R-L F   | 23.70   | 17.00  | 21.44  | 21.33   | 21.44      | 16.98  | 19.01   | 20.78         |
| mT5 In-language                                                        | R-1 F   | 44.26   | 35.06  | 39.41  | 43.81   | 41.00      | 32.26  | 4.27    | 40.04         |
|                                                                        | R-2 F   | 21.73   | 13.63  | 17.76  | 19.29   | 20.22      | 14.34  | 0.58    | 18.23         |
|                                                                        | R-L F   | 24.25   | 17.53  | 22.03  | 23.76   | 24.44      | 18.67  | 3.06    | 21.93         |
| Abstractive Summarisation after mBERT Pre-Abstractive Extractive Step  |
| mT5 Cross-lingual zero-shot transfer                                   | R-1 F   | 37.24   | 7.19   | 9.14   | 10.02   | 9.56       | 6.30   | 12.40   | 17.08         |
|                                                                        | R-2 F   | 13.00   | 1.68   | 1.87   | 2.48    | 2.27       | 1.30   | 2.82    | 5.25          |
|                                                                        | R-L F   | 19.68   | 5.08   | 5.97   | 6.89    | 6.74       | 4.58   | 7.37    | 10.00         |
| mT5 In-language multi-task                                             | R-1 F   | 35.56   | 27.05  | 32.59  | 32.94   | 30.01      | 23.53  | 32.90   | 31.30         |
|                                                                        | R-2 F   | 12.28   | 7.84   | 13.06  | 11.65   | 13.14      | 9.37   | 10.24   | 11.24         |
|                                                                        | R-L F   | 18.70   | 13.71  | 18.93  | 18.16   | 18.82      | 14.22  | 16.93   | 17.25         |
| mT5 In-language                                                        | R-1 F   | 37.24   | 29.65  | 36.02  | 39.79   | 37.21      | 28.47  | 4.32    | 35.03         |
|                                                                        | R-2 F   | 13.00   | 8.32   | 14.08  | 13.86   | 15.46      | 10.66  | 0.10    | 12.37         |
|                                                                        | R-L F   | 19.68   | 14.76  | 20.08  | 21.17   | 13.20      | 16.65  | 2.80    | 18.04         |

> ROUGE F-measure results of the three evaluations presented in the paper on WikinewsSum. We compare the extractive models, and mT5 in the three training scenarios and with two different pre-abstractive extractive steps (Oracle and mBERT) for each language of the WikinewsSum dataset in addiction to the all dataset.

### BERTScore scores

| Methods                                                                | Metrics | English | German | French | Spanish | Portuguese | Polish | Italian | All Languages |
| ---------------------------------------------------------------------- | ------- | ------- | ------ | ------ | ------- | ---------- | ------ | ------- | ------------- |
| Extractive Summarisation                                               |
| DistilmBERT                                                            | B-S P   | 0.6920  | 0.6669 | 0.6357 | 0.6807  | 0.6680     | 0.6455 | 0.6706  | 0.6697        |
|                                                                        | B-S R   | 0.7196  | 0.6890 | 0.6846 | 0.7104  | 0.7084     | 0.6834 | 0.7068  | 0.7021        |
|                                                                        | B-S F   | 0.7052  | 0.6774 | 0.6585 | 0.6949  | 0.6869     | 0.6633 | 0.6879  | 0.6850        |
| mBERT                                                                  | B-S P   | 0.6908  | 0.6679 | 0.6354 | 0.6810  | 0.6673     | 0.6456 | 0.6618  | 0.6695        |
|                                                                        | B-S R   | 0.7215  | 0.6931 | 0.6855 | 0.7124  | 0.7084     | 0.6848 | 0.7033  | 0.7041        |
|                                                                        | B-S F   | 0.7055  | 0.6799 | 0.6587 | 0.6960  | 0.6865     | 0.6640 | 0.6816  | 0.6859        |
| XLM-RoBERTa                                                            | B-S P   | 0.6900  | 0.6658 | 0.6351 | 0.6794  | 0.6660     | 0.6451 | 0.6752  | 0.6684        |
|                                                                        | B-S R   | 0.7173  | 0.6878 | 0.6834 | 0.7087  | 0.7061     | 0.6831 | 0.7099  | 0.7005        |
|                                                                        | B-S F   | 0.7031  | 0.6762 | 0.6576 | 0.6934  | 0.6848     | 0.6629 | 0.6917  | 0.6836        |
| Oracle                                                                 | B-S P   | 0.7238  | 0.6947 | 0.6528 | 0.7058  | 0.6930     | 0.6638 | 0.6919  | 0.6955        |
|                                                                        | B-S R   | 0.7436  | 0.7144 | 0.6967 | 0.7228  | 0.7266     | 0.7024 | 0.7190  | 0.7217        |
|                                                                        | B-S F   | 0.7332  | 0.7039 | 0.6731 | 0.7138  | 0.7087     | 0.6818 | 0.7047  | 0.7077        |
| Abstractive Summarisation after Oracle Pre-Abstractive Extractive Step |
| mT5 Cross-lingual zero-shot transfer                                   | B-S P   | 0.7526  | 0.6814 | 0.6687 | 0.7014  | 0.6864     | 0.6468 | 0.6820  | 0.7009        |
|                                                                        | B-S R   | 0.7199  | 0.6431 | 0.6579 | 0.6650  | 0.6641     | 0.6218 | 0.6480  | 0.6717        |
|                                                                        | B-S F   | 0.7354  | 0.6614 | 0.6627 | 0.6824  | 0.6746     | 0.6337 | 0.6644  | 0.6855        |
| mT5 In-language multi-task                                             | B-S P   | 0.7494  | 0.7219 | 0.7130 | 0.7306  | 0.7274     | 0.6887 | 0.7203  | 0.7274        |
|                                                                        | B-S R   | 0.7190  | 0.6937 | 0.7174 | 0.7030  | 0.7140     | 0.6847 | 0.6942  | 0.7074        |
|                                                                        | B-S F   | 0.7334  | 0.7070 | 0.7138 | 0.7161  | 0.7197     | 0.6857 | 0.7066  | 0.7165        |
| mT5 In-language                                                        | B-S P   | 0.7526  | 0.7264 | 0.7164 | 0.7374  | 0.7381     | 0.6974 | 0.4603  | 0.7321        |
|                                                                        | B-S R   | 0.7199  | 0.6939 | 0.7179 | 0.7073  | 0.7194     | 0.6908 | 0.5261  | 0.7092        |
|                                                                        | B-S F   | 0.7354  | 0.7093 | 0.7153 | 0.7216  | 0.7277     | 0.6931 | 0.4905  | 0.7196        |
| Abstractive Summarisation after mBERT Pre-Abstractive Extractive Step  |
| mT5 Cross-lingual zero-shot transfer                                   | B-S P   | 0.7202  | 0.6680 | 0.6571 | 0.6858  | 0.6757     | 0.6412 | 0.6693  | 0.6828        |
|                                                                        | B-S R   | 0.7004  | 0.6363 | 0.6517 | 0.6576  | 0.6586     | 0.6180 | 0.6459  | 0.6615        |
|                                                                        | B-S F   | 0.7098  | 0.6515 | 0.6538 | 0.6712  | 0.6666     | 0.6290 | 0.6572  | 0.6716        |
| mT5 In-language multi-task                                             | B-S P   | 0.7157  | 0.6958 | 0.6953 | 0.7069  | 0.7094     | 0.6700 | 0.7045  | 0.7022        |
|                                                                        | B-S R   | 0.6981  | 0.6774 | 0.7033 | 0.6891  | 0.7011     | 0.6702 | 0.6869  | 0.6910        |
|                                                                        | B-S F   | 0.7065  | 0.6861 | 0.6982 | 0.6976  | 0.7046     | 0.6693 | 0.6952  | 0.6960        |
| mT5 In-language                                                        | B-S P   | 0.7202  | 0.7043 | 0.7020 | 0.7151  | 0.7186     | 0.6836 | 0.4495  | 0.7091        |
|                                                                        | B-S R   | 0.7004  | 0.6807 | 0.7069 | 0.6948  | 0.7064     | 0.6803 | 0.5213  | 0.6949        |
|                                                                        | B-S F   | 0.7098  | 0.6919 | 0.7026 | 0.7044  | 0.7116     | 0.6811 | 0.4822  | 0.7012        |

> BERTScore (Zhang et al., 2020b) precision (B-S P), recall (B-S R), and F1 (B-S F) results of the three evaluations presented in the paper on WikinewsSum. We compare the extractive models, and mT5 in the three training scenarios and with two different pre-abstractive extractive steps (Oracle and mBERT) for each language of the WikinewsSum dataset in addiction to the all dataset. Hash code for the BERTScore metric: bert-base-multilingual-cased_L9_no-idf_version=0.3.11(hug_trans=4.13.0)\_fast-tokenizer

## Usage

The `mdmls` pip package allows to run the combination of an extractive method combined with an abstractive one.

```bash
pip install mdmls
```

It can be used as follows in Python.

```python
from mdmls import Summarizer

sum = Summarizer()
summary = sum(LONG_TEXT_TO_SUMMARIZE)
```

Or directly using the CLI.

```bash
mdmls "LONG_TEXT_TO_SUMMARIZE"
```
