# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:27:57 2022
Edited on Fri Oct 7

@title Keras MNIST Example
@author: TEastman
@author Lucas Kohorst
"""

"""
TODO:
  - [ ] Add in scalars for monitoring
  - [ ] Add in templates for reporting out metrics and plots
  - [ ] Add in optimization
  - [ ] Scaffold for data ingestion
  - [x] Add in basic pipeline
  - [x] Maybe add in artifacts?
  - [x] Take this and create into a python class to be used
"""

from mimetypes import init
from clearml.automation.controller import PipelineDecorator
from clearml import Task, TaskTypes, Logger
import matplotlib.pyplot as plt
import argparse
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
from dataclasses import dataclass
#pip install transformers
# Variables for Pipeline Decorators
project = "keras_example"
pipeline = "Keras Example Pipeline"
version = "0.0.1"

class BittensorTraining:
  def __init__(self, args):
    print("in init")
    self.epochs = args.epochs
    self.steps = args.steps
    print("end of init")



#############################################STARTS OF TRAINING CODE###########################################################################

#no need of component
#for reuse of code. same val and train data for next run too.
  def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

  set_seed(1)


  @PipelineDecorator.component(return_values=['x_train', 'y_train', 'x_val', 'y_val','target_names'], cache=True, task_type=TaskTypes.data_processing, execution_queue="training")
  def preprocess_data(self):
    """
    preprocess test data

    :return train_dataset: preprocessed datapoints for a dataset to train
    :return val_dataset: preprocessed datapoints for a model to be validated against
    """
    import torch
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    import numpy as np
    import random
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from torchvision import transforms


    model_name = "bert-base-uncased"

    ####bert config just for reference
    '''
    ( vocab_size = 30522hidden_size = 768num_hidden_layers = 12num_attention_heads = 12
    intermediate_size = 3072hidden_act = 'gelu'hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1max_position_embeddings = 512
    type_vocab_size = 2initializer_range = 0.02layer_norm_eps = 1e-12pad_token_id = 0
    position_embedding_type = 'absolute'use_cache = Trueclassifier_dropout = None**kwargs )
    '''

    max_length = 512

    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    test_size=0.2

    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target
    (train_texts, valid_texts, train_labels, valid_labels), target_names = train_test_split(documents, labels, test_size=test_size), dataset.target_names
    x_train, y_train, x_val, y_val = train_texts, valid_texts, train_labels, valid_labels
    return x_train, y_train, x_val, y_val, target_names

  train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
  valid_encodings = tokenizer(y_train, truncation=True, padding=True, max_length=max_length)


######################DATALOADER########################
  class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
  train_dataset = NewsGroupsDataset(train_encodings, train_labels)
  valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

  model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")

  from sklearn.metrics import accuracy_score

  @PipelineDecorator.component(return_values=['acc'], cache=True, task_type=TaskTypes.qc, execution_queue="training")
  def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #     calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

  # Only when a return value is used, the pipeline logic will wait for the component execution to complete
  @PipelineDecorator.component(return_values=['trainer'], cache=True, task_type=TaskTypes.training, execution_queue="training")
  def train_model(train_dataset, valid_dataset):

    ##################################Train ARGUMENTS###############################
    training_args = TrainingArguments(
    output_dir='./results',

    ################change number of epochs for testing to 3 , for actual training 1000
    num_train_epochs=1000,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=12,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=0.0001,
    logging_dir='./logs',
    load_best_model_at_end=True,

    logging_steps=400,
    save_steps=400,
    evaluation_strategy="steps",
    )
    #################################################################################

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer

    ####Train
  train_model(train_dataset, valid_dataset).train()

  train_model(train_dataset, valid_dataset).evaluate()

    #####bin or pt file - used to serve in bittensor serving
  model_path = "bert_train"
  model.save_pretrained(model_path)
  tokenizer.save_pretrained(model_path)
  torch.save(model.state_dict(), 'bert-model.bin') #can remove if throws error



#############################################END OF TRAINING CODE###########################################################################

  @PipelineDecorator.component(return_values=[], cache=True, task_type=TaskTypes.monitor, execution_queue="serving")
  def serve(self):
      import bittensor

      print('serve')

  @PipelineDecorator.component(return_values=[], cache=True, task_type=TaskTypes.monitor, execution_queue="training")
  def report_metrics(self, history):
    """
    report metrics on the training

    :param history: the history of the training from the model
    """
    # gathering metrics of the model
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) +1)

    # plotting data on the model
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

  # Pipeline execution context
  @PipelineDecorator.pipeline(name=pipeline, project=project, version=version, add_pipeline_tags=True, pipeline_execution_queue="training")
  def executing_pipeline(self):
    """
    execute the pipeline
    """

    # process/load the data
    print("loading data")
    self.serve()
    # (x_train, y_train), (x_val, y_val) = self.preprocess_data()
    print("data loaded")

    # # create the datasets
    # train_dataset = self.create_dataset(x_train, y_train)
    # val_dataset = self.create_dataset(x_val, y_val)

    # # train the model
    # history = self.train_model(train_dataset, val_dataset)

    # report out metrics on the training
    # TODO: metrics are in progress
    # self.report_metrics(history)

    # self.serve()

if __name__ == '__main__':
  # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
  # PipelineDecorator.set_default_execution_queue('training')
  # Run the pipeline steps as subprocesses on the current machine, great for local executions
  # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
  PipelineDecorator.run_locally()

  description = 'Keras MNIST Example'
  # defining command line parameters
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--steps', type=int, default=200, help='steps per epoch for training (default: 200)')
  parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
  # parser.add_argument('--project', type=str, default="keras_example", help='name of the project (default: keras_example)')
  # parser.add_argument('--pipeline', type=str, default="Keras Example Pipeline", help='name of the pipeline (default: Keras Example Pipeline)')
  # parser.add_argument('--version', type=str, default="0.0.1", help='version of the project (default: 0.0.1)')

  # parsing the arguments
  args = parser.parse_args()

  BittensorTraining(args=args).executing_pipeline()
