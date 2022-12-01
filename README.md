# 365-DataScience-Machine Learning Model & Tableau Dashboard
# Notes:

1. Export the given original data to MySQL tables and create indexes etc.
2. Create Views and Functions that can serve the dashboard and ML model requirements
3. Export data from created view to Jupyter Notebook
4. Preprocessing of data in Pandas; Added columns that give data on courses watched using pandas 
5. Complete the machine learning models in Jupyter Notebook
6. Install tabpy server (local host) - tabpy is not working with Python 3.9 while Anaconda3 defaults with Python 3.9
7. Create separate conda environment with python 3.7 and install all libraries including sklearn / tensorflow
8. Install and start tabpy server in the above envionment
9. The required data - the final table exported at step 4 (along with few other tables) will be sufficient - can be accessed directly by creating an MySQL data source. I have instead exported the data as csv files.
10. Create charts in tableau
11. Publish the charts to Tableau-public / Tableau Online
12. Create an tabpy analytics extension connection in tableau
13. Install tabpy in Jupyter notebook and connect through a client
14. Write python functions that uses the ML models for prediction
15. deploy the functions to tabpy server
16. Write python scripts in tableau that queries the above models through tabpy extension

## Some restrictions / problems / limitations / insights:

1. Tabpy server could not be deployed in Heroku - it looks like Tabpy can not be accessed for remote deployment
2. There does not seem to be any other free tabpy deployment on the web currently possible
3. The tableau dashboard that uses Tabpy analytics extension cannot be published in tableau public or online without tabpy being deployed in web
4. There are version mismatch problems - in installing tensorflow / tabpy / pickle etc. I reinstalled all required software in new conda environment with python 3.7
5. Tableau does not allow passing rows of data to tabpy / ML models. Instead each column value needs to be specified as separate arguments
6. It should be possible for Tableau to access test data table and give predictions for each row while the chart accesses any selected columns from the row for view. Does not seem to be currently possible
7. I found it more convenient to use MySQL client to do most of the preprocessing before exporting the data for model building. However, it can also be done from Jupyter using PySQL
8. Imbalance techniques tend to overfit the data. The accuracies for target label went down when imbalance techniques are applied. The overall accuracy and accuracy of majority class label only increases
9. Tensorflow could not be used to train the accuracy measure for minority label - the focus is to miminize the overall loss objective function. Using GridSearch however allows giving a scoring parameter (like precision) on which performance tuning could be done as against loss function
10. Autoencoder model gives accuracies that either favour precision or recall. The overall low accuracies given by Autoencoders imply additional data is required in this problem
11. A dashboard should ideally give insights, not just on static processed data, but also on live predictions given by machine learning model. The above dashboard gives a chart that provides predictions for sample test data (this data can also be live connected and filtered)
