# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

A random forst classifier ist used to predict if the salary of a person is lower or higher than 50K.

## Intended Use

The [model](./model/model.pkl) should be used to predict the category of the person's salary _(lower/higher 50K)_.

## Training Data

The [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) dataset is used. 80% of the data is used for the training.

## Evaluation Data

The [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) dataset is used. 20% of the data is used for the evaluation.

## Metrics

The model was evaluated with different metrics. The scores are listed in the table below:

| Metric    | Score              |
|-----------|--------------------|
| Precision | 0.7501909854851031 | 
| Recall    | 0.6278772378516624 | 
| F-Beta    | 0.6836059867734076 | 

## Ethical Considerations

The model performance was calculated on data slices. The model may bias people on profession or gender. PLEASE DO NOT USE THIS MODEL IN PRODUCTION, it is recommended for educational purpose only.

## Caveats and Recommendations

Feature biases need further investigation before using the model in production.
