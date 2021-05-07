# jobScrappingAIProject
The details of the files are given below.

1) finaljobsearch.py - This python file does the web scrapping job from the https://jobs.sap.com.
It captures various job details like title, job description, date, location, requisition id, 
work area, job segment.

The "Work Area" is extracted from the job page and it is also stored in csv file.

All the extracted details are stored in FinalDataset.csv. A bit of cleaning is done while
extracting and storing the information in csv. Additional cleaning will be done during
data processing for the implementation of the classification model.

2) finalcutmodel.py - This python file does the data-processing and implementation of 
classification model. A neural network (MLP model) is used to perform the classification. Tensorflow and Keras
frameworks and scikit-learn library are used for implementation.

The classification metrics - recall, precision, F1 Score are stored in ClassificationReport.csv

The model is performing with 87% training accuracy and around 60% testing accuracy.

In the interest of time appropriate regularization is applied for improving the performance.
Data preprocessing and regularization can be further used to improve the performance which
can be done after discussion with domain expert of the job portal or users of job portal.
LSTM or BERT with transformers are ideal for text processing, it can used based on 
discussion.

3) FinalDataset.csv - The dataset extracted from job portal for all jobs.

4) ClassificationReport.csv - The table of classification metrics - recall, precision, F1 Score
