# BERT

This repository contains an implementation of BERT fine-tuning for Multi-label classification. In this case, the goal is to classify a document into one or more classes/labels. It is a generalization of the multiclass classification problem and the same implementation can be used for it. This implementation is based on [BERT](https://arxiv.org/pdf/1810.04805.pdf).

## How to use?

The usage is similar to BERT for other fine-tuning implementations, such as described in [BERT's repository](https://github.com/google-research/bert). To train a model *run_classifier.py* can be called:

```
python run_classifier.py --task_name=$TASK \
--do_train=true \
--do_predict=false \
--data_dir=$DATASET_DIR \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=.$BERT_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=10.0 \
--output_dir=$OUTPUT_DIR \
--add_dense=false
```
To predict samples, *run_classifier.py* can be called in the following way:

```
python run_classifier.py --task_name=$TASK \
--do_train=false \
--do_predict=true \
--data_dir=$DATASET_DIR \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=.$BERT_DIR/bert_model.ckpt \
--max_seq_length=128 \
--output_dir=$OUTPUT_DIR \
--add_dense=false\
--cut_off=0.5 \
--cutoff_type=static
```

In both cases, *--task_name* represents the task to be executed: the referred Processor class will be called if exists. The *--add_dense* parameter adds a dense layer to the logits from BERT's output layer. The final activation used is a sigmoid function, such that the final outputs consist on a vector, in which a position *i* in this vector represents a probability of belong to a class *i*. For this matter, the parameter *--cut-off* is the cut-off probability to belong to a class and the cut-off can be applied statically, via the *cut_off* parameter, or dynamically, learned from an dev set: the *--cutoff_type* can receive, then, *static* or *dynamic*. If no value is passed to *--cutoff_type*, then the predictor will assume the probabilities is the desired output.

To test this method I used four datasets available in the literature: 

* **Movie Lens**: ML100k dataset from [GroupLens](https://grouplens.org/datasets/movielens/). The title and summary of a movie to predict the genres of a movie.
* **SE0714**: SE0714 dataset from [Deepmoji](https://arxiv.org/pdf/1708.00524.pdf). The text is used to predict the emoji related to the text. 
* **PsychExpEmoji**: SE0714 dataset from [Deepmoji](https://arxiv.org/pdf/1708.00524.pdf). The text is used to predict the emoji related to the text. 
* **Toxic**: Toxic Comments from [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).  Dataset used to predict the class of toxic comments in the text. For Toxic, only the probabilities of a commentary belonging to each class is predicted

The implementation was tested on these datasets. And the best results obtained on them are shown below. We don't show each individual class due to space limitations. For dynamic cut-off type, the dataset was divided in train, dev and test sets. For all cases BERT-Base is used to train and test.

| Dataset | BERT pre-trained model used  | Max Length | Training Epochs | Cut-off type |Add Dense-layer? | AUC | Hamming Loss | F1 |
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |
| Movie Lens | BERT-Base Uncased   | 128  | 4 | Dynamic  | No |0.895536278 |	0.092306414 |	0.672362743 |
| SE0714 | BERT-Base Uncased   | 128  | 4  | Dynamic | Yes |0.934132073 |	0.057333333 |	0.678547981 |
| PsychExpEmoji | BERT-Base Uncased   | 140  | 4 | Dynamic  | Yes | 0.922672041 | 0.080309396	| 0.732810139 |
| Toxic | BERT-Base Uncased  | 140  | 4  | Dynamic | Yes | 0.98606 | NA	| NA |
