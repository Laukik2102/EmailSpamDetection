# EmailSpamDetection

Objective

To detect whether an Email is Ham or Spam using Naive Bayes Classifier Algorithm.

Science and Engineering problem motivating the project

With the growth of technology and IT industry, the advertisement industry also grew, and it benefitted from the technology as much as it could. While we hold the magical door to information in our hands using our cell phones and laptops, this is also a gateway to a security breach to our personal information. Spammers took this opportunity as a golden chance to lure the average user of technology into believing things and advertisement that they shouldn't and bait them with attractive offers via text messages and emails. These spam emails mostly contain offers that look too good to be true and people unknowingly believed in those offers and suffered from the loss of money and personal information.
        	
This problem inspired the computer scientist of the last century to come up with a solution that could save people from falling into this trap. The proposed solution was to classify the messages as spam or not spam and then filtering spam messages from inbox, saving the time for the user. To implement this, several methods are present but for the sake of this project, Naive Bayes has been chosen the reason of which will be explained later.

1.  Importing the dataset

We are doing this project in Python environment. So, using read_csv() function, we imported our dataset into python. Description of our dataset is given below:- 
As email ids have become such a essential part of our lives, that it has become an accustomed  habit to keep checking for every mail now and then and sorting out which mail is important and which isn’t. In such a busy life promotional or spam emails are a waste of time and  memory. One needs to know and understand as to which email matters to them and which is wasteful.
Here we are considering a dataset that has 5574 rows of spam and ham messages combined. It consists of predefined messages which tell us whether a message is spam or ham. 

2. Exploratory Analysis

Exploratory Analysis gives us a complete description of the data and tells us how this data can be utilized accurately in a proper format for analysis purposes. We found out the mean, median and various important aspect of our data before starting with the algorithmic implementation. Also, we have plotted graphs for data visualization purposes. As we continue our analysis, we start thinking about the features that we are going to be working with. This goes along with the idea of Feature Engineering. It is a very large part of spam detection. The better our domain knowledge on the data, the better our ability to engineer more features from it.	

3. Text pre-processing

The main issue with our data is that it is all in text format (string). To perform classification task, we require numerical feature vector. We have converted our string data into vector form by ‘Bag-of-words’ technique. Here, each unique word of a text would be represented by one number. There are 3 main steps of performing this:

a.	Firstly, we converted string data into list of words.
b.	Then by tokenize method, converted that list of words into list of tokens. Here, all the common words such as: ‘the’, ‘a’, ‘etc’ are removed as they are insignificant for implementation part.
c.	Finally, we converted list of tokens into vector format so that scikit learn’s algorithm model can work by ‘Bag-of-Words’ model.

4. Training the Model 

After the text is preprocessed and various insights about data has been obtained, we trained our model using Naive Bayes classifier algorithm. Firstly, we divided our dataset into training and testing data and then our model was able to predict whether a given email message was Ham or Spam. For eg, if we consider the third message in our dataset:
 “ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...“ Our model gives the output as follows:

 
 
We already know that this data was a Ham. After implementing the model, it gave us the predicted output also as Ham which means that our model has been trained effectively and is giving us accurate results. 
