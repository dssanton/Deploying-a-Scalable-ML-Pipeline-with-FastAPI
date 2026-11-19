# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses a Random Forest Classifier from sklearn to predict whether someone earns more than $50k annually. The model uses data from the Census Income dataset. I used 100 trees with a random seed of 42. 

## Intended Use
This model is for my D501 project 2. The goal is to build and test a classification model and make its predictions available via FastAPI. It predicts whether someone makes $50k income and more or less than $50k. This is for WGU. 

## Training Data
I used the Census Income Dataset included. It has information about folks in the USA - age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and salary. I split the data 80/20 using 80% for training and 20% for testing. The categorical features were converted to numeric using one-hot encoding. 

## Evaluation Data
I evaluated the model against the 20% of the dataset which was split off. I used the same encoders to process the test data. I also evaluated categorical values to see how the model performs across different groups. They're saved in slice_output.txt. 

## Metrics
Precision: 0.7457 | Recall: 0.6404 | F1: 0.6890

The precision was 74.57%, meaning around 3 out of 4 people predicted to earn more than $50k income were classified correctly. Recall was 64.04%, meaning that the model missed around 36% of the people who actually earned more than $50k income. The F1 score was 68.9%, showing the balance between precision and recall.

## Ethical Considerations
The dataset has race, sex, marital status, and other demographic info. These can highlight economic inequalities, so the model might be biased for different groups. 

## Caveats and Recommendations
$50k income isn't that high anymore. It's not quite a meaningful number today, especially since the dataset is old. The recall could be better. I'd want to try it with newer data. 
