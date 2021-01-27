# flask_example_hate_speech
A repository containing an example of using the Flask framework to deploy a Machine Learning model in a web application.
The model used is just for the sake of demonstration. It is a logistic regression on top of TFIDF transformation over 1000 dimensions(nothing really great for such a task). 
So the idea here is to use the flask framework and not to have a great model to predict hate speech. If you are interested in such a task, contact me ! 


### Step 1 : Basic form where you can add the text you want to analyze 
![alt text](https://github.com/kimakour/flask_example_hate_speech/blob/main/images/first.PNG)
### Step 2 : you add the text and push the predict button 
![alt text](https://github.com/kimakour/flask_example_hate_speech/blob/main/images/second.PNG)
### Step 3 : You get the prediction and you can once more add a text you want to analyze 
![alt text](https://github.com/kimakour/flask_example_hate_speech/blob/main/images/third.PNG)



work based on these two articles : 
- https://www.analyticsvidhya.com/blog/2020/04/how-to-deploy-machine-learning-model-flask/ 
- https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99 
