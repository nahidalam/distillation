Work in progress implementation of https://arxiv.org/pdf/1903.12136.pdf

### Datasets
- Download SST-2 for single sentence classification task
- Quora QQP for sentence pair classification task

### Architecture
1. Embedding - create embedding using word2vec
2. Teacher Model
  - define the model (using pretrained BERT so just import it)
  - train the finetuned model (using respective dataset)
  - output logits
  - Train for 4 different lr and find the best one
3. Student Model
  - input data in proper format
  - define the custom student model
  - train it with the loss function suggested by the paper
  - make sure you use the right hyperparameter

4. Compute Metrics [TO DO]
  - accuracy, F1 score etc.
