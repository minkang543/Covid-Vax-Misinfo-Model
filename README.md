# Covid-Vax-Misinfo-Model

## Research Question 
What are the dominant themes in Covid-19 vaccine misinformation on Twitter and are there trends in talking points over the course of the pandemic?

## Research Project Title
Observing Patterns in Covid-19 Vaccine Misinformation on Twitter.

## Introduction and Thesis
The issue of vaccine hesitancy has always been a challenge in widespread distribution. With the
COVID-19 pandemic, came the rise of a misinformation campaign against vaccines for the
coronavirus. While anti-vaccination attitudes have always existed, the prevalence of social
media during a pandemic has allowed anti-vaccination content to be endlessly available. We
believe that vague, difficult to understand topics regarding vaccines have always been
spotlighted and exploited against the general public. We will examine the Twitter Data about
COVID-19 created by Banda et. Al., specifically its vaccine over the course of the pandemic and
see how attitudes have shifted. We will use the BERT (bidirectional encoder representations
from transformers) model which has achieved excellent performance in the classification of antivaccination
tweets done by Quyen G. To et. Al. Our work aims to both classify misinformation
and examine the trends over the course of the pandemic. Previous works have focused on only
classification and generally focus on concurrent topics of misinformation. By comparing the
relevancy of certain topics over time, we hope to find trends in the direction of anti-vaccine
rhetoric. By understanding this, measures can be taken to predict future topics and stop future
outbreaks of misinformation. Furthermore, we aim to use our findings and compare it to the
concurrent adult vaccination rate. This can give insight into the effectiveness of certain topics of
misinformation and see how rates of vaccination compare next to trending sources of antivaccine
information. This will allow us to categorize topics by their harmfulness and it could be
used to direct resources to combat the worst offenders.

## Related work
Using machine learning to detect misinformation has been explored extensively and has been
used to combat the recent trend of vaccine misinformation. Quyen G. To et. Al compares
different natural language machine learning models to see which is the most suitable one for
identifying anti-vaccination tweets. They examined bidirectional encoder representations from
transformers (BERT) and the bidirectional long short-term memory networks with pre-trained
GLoVe embeddings (Bi-LSTM) with classic machine learning methods including support vector
machine (SVM) and naïve Bayes (NB). They found that the BERT model outperformed the
others with excellent performance and is suitable to identify anti-vaccination tweets in future
studies and will be used in this one. However, rather than focusing on the classification of
misinformation, we aim to use machine learning to both classify and sort different topics of
misinformation.
Maxwell A. Weinzierl and Sanda M. Harabagiu tackle the automatic detection of specifically
known COVID-19 vaccine misinformation. They introduce CoVaxLabs, a dataset of tweets
judged relevant to many different topics of misinformation and charts the dataset in a
Misinformation Knowledge Graph. However, rather than compare the topics of misinformation
and how they relate in general, we will see how trends of misinformation has changed over the
course of the pandemic.
Hayawi, K., et al created the dataset ANTi-Vax, a Twitter dataset for COVID-19 vaccine
misinformation detection. They also found that BERT was the best performer in classification
and concluded that Machine learning-based models are effective in detecting misinformation.
Once again, we look to classify data over the period of the pandemic so far to search for trends
in misinformation.
Using machine learning to classify misinformation is not anything new. However, previous works
attempt to prove the model’s success and its effectiveness in classification. We believe machine
learning models have been sufficiently shown to excel in this context and should be used to find
data/trends unavailable to other models. Social media, specifically the Twitter API, is an ocean
of data largely untapped that could give major insight into public opinion about current events. It
could signify attitudes about certain topics and we use it hear to try to understand public
attitudes about the coronavirus vaccine and how they have shifted over time.

## Methods
### Data
We will use the CoVaxLies dataset from Weinzierl and Harabagiu. This gives us a recently
reviewed dataset that is broken up into different Misinformation Targets. These targets were
obtained from en.wikipedia.org/wiki/COVID-19_misinformation#Vaccines which also provides
citations to scientific articles to debunk the misinformation. This dataset was particularly chosen
because it is a relatively recent dataset that also categorizes the misinformation regarding the
vaccine, while other works have classified whether or not misinformation exists. Using the
Twitter API, we will obtain the text body of the tweet from the ID supplied by CoVaxLies and use
the classifications to train our model.
Their data was separated into a training set, a test set, and a development set. Each entry came
with the ID of the tweet, the misinformation target, and whether the tweet was relevant to the
topic or not. Using the Twitter API, we retrieved the body for each tweet. Unfortunately, many of
the tweets used were lost or deleted since their study, resulting in a training set of 1114 tweets
compared to the previous 3735 tweets. The test set was reduced to 333 tweets from 1038 and
the development set became 131 tweets from 415. We will interpret the data using Word-Piece
Tokenization to process the tweets to produce tokens. This allows us to ensure every word is
able to be tokenized into interpretable language by breaking down complicated language into
common sub-word components.
After the model is trained, we will once again use the Twitter API to get Twitter Data over the
course of the pandemic. We connect to the Full Achieve Search end point and use the query
(covid OR coronavirus) vaccine -is:retweet. This searches for "covid vaccine” or “coronavirus
vaccine” while parsing out retweets from the search. We use this specific query as it is identical
to the query used by the CoVaxLies dataset to use our model to classify data consistent with the
training data. We collected 4000 tweets from every month starting from March 2020 all the way
to December 2021.
However, the Twitter API gathers data starting from end of the range of time given. Since only
500 tweets are able to be gathered with each end point connection, we will change the date
search range for each end point connection. For instance, while searching through the month of
April 2020, to get 4000 tweets, the end point must be contacted 8 times. Accordingly, we will
separate the month into 8 sections and search each accordingly to get an even spread of data.
We will call this dataset the MonthToMonth data set from now on.
Using the MonthToMonth dataset, we used our trained model to classify the tweets and find the
frequency of each misinformation topic according to the month.
By comparing this data with the vaccination rates from https://ourworldindata.org/covidvaccinations?
country=USA, we can see how effective certain topics of misinformation are
compared with others. Meaning, if we see the rate in change of vaccination rate lower after
seeing the topic of “The COVID-19 vaccine contains tissue from aborted fetuses” trend, we can
assume that this topic was particularly persuasive in dissuading the public to be innoculated.
### Misinformation Targets
There are a total of 17 topics of misinformation we will be targeting in this paper. These topics
were chosen from Maxwell A. Weinzierl and Sanda M. Harabagiu in their work, Automatic
detection of COVID-19 vaccine misinformation with graph link prediction. They chose the topics
by taking advantage of existing efforts of pinpointing misconceptions about the COVID-19-
vaccines from organizations such as Wikipedia, UC Davis Health, Mayo Clinic, University of
Missouri Health Care, University of Alabama at Birmingham, and PublicHealth.org. In their
paper, they named these 17 topics as Misinformation Targets (MisTs) and we will call them so
from now on.
Here, we decided to use the selections of MisTs from this previous work as they created the
dataset, CoVaxLies, with a training, development, and testing already classified with these 17
MisTs by researchers from the Human Language Technology Research Institute at the
University of Texas. As we lacked the resources to pre-classify a large dataset like this, we
decided to take advantage of this previous work as it aligns heavily with what we are looking for.
However, we bring novelty as in their paper, they applied their model to a large number tweets
in one large dataset, we aim to find the frequency of these different topics over the course of
time. We believe this adds sufficient novelty as it would give an understanding of what different
topics of misinformation was being discussed at different times. By comparing this with the
inoculation rates of the population, this gives insight into how misinformation could directly affect
vaccination rates, while the work by Weinzierl focused on seeing how the different MisTs
compared with one another on a Misinformation Knowledge Graph.
### Models
We will use a model of BERT called COVID-19-Language Model COVID-Twitter-BERT-v2 made
by M. Müller, M. Salathe, and P.E. Kummervold. This is a model that was pre-trained with 97
million COVID-19 tweets and has been shown to have higher performance than the standard
model regarding language surrounding COVID-19.
Using CoVaxLies, we fine tune the model to be able to classify the different MisTs from the body
of a tweet. First, we preprocess our data through an auto-tokenizer from Hugging Face’s
Transformers class. This breaks down our data into a list of integers, i.e. [101, 2926, 2007,
2023, 3563, 17404…], each integer representing a token. This set of integers could also be
decoded to reobtain our initial text. After our data is tokenized, we feed the data to the already
trained CT-BERT model.
After our model is sufficiently fine-tuned, we apply the model on our MonthToMonth dataset,
keeping track of the dates and months of each Tweet. By graphing the frequency of different
topics of misinformation over time, we hope to see trends in different MisTs. Using the
innoculation rates from https://ourworldindata.org/covid-vaccinations?country=USA, we hope to
find a link between trending MisTs and a drop-in inoculation rate. This would show how the
propagation of misinformation on social media is actively decreasing vaccination intent.
Furthermore, it would show how certain MisTs are more effective in doing so.

## Results
We hope to produce a precise model that is able to identify what kind of misinformation is in a
Tweet and when it occurred. Then, we will find the most prevalent sources of misinformation
towards vaccination rates in the USA.

## Discussion
This work may be limited by it not being able to work with novel sources of misinformation.
Furthermore, our model will not address the stance for the Tweet.

## Conclusion
Vaccination rates, while not what not we had hoped for, are improving everyday. With more and
more data coming out regarding the efficacy of the vaccine, it's hard to prove that the vaccine
does not work. However, misinformation is an issue that will never go away and continues to
plague a segment of our population. By finding the most harmful sources of misinformation,
resources could be targeted stopping their spread and find trends in what may be the next new
reason not to be vaccinated.

## References
To, Quyen G., et al. “Applying Machine Learning to Identify Anti-Vaccination Tweets during the
Covid-19 Pandemic.” International Journal of Environmental Research and Public
Health, vol. 18, no. 8, 2021, p. 4069., https://doi.org/10.3390/ijerph18084069.
Weinzierl, Maxwell A., and Sanda M. Harabagiu. “Automatic Detection of COVID-19 Vaccine
Misinformation with Graph Link Prediction.” Journal of Biomedical Informatics, vol. 124,
2021, p. 103955., https://doi.org/10.1016/j.jbi.2021.103955.

M. Müller, M. Salathe, P.E. Kummervold, Covid-twitter-bert: A natural language
processing model to analyse covid-19 content on twitter, https://arxiv.org/abs/
2005.07503.

Hayawi, K., et al. “ANTi-Vax: A Novel Twitter Dataset for COVID-19 Vaccine Misinformation
Detection.” Public Health, vol. 203, 2022, pp. 23–30.,
https://doi.org/10.1016/j.puhe.2021.11.022.
S. Loomba, A. de Figueiredo, S.J. Piatek, K. de Graaf, H.J. Larson, Measuring the
impact of covid-19 vaccine misinformation on vaccination intent in the uk and usa,
Nature Human Behaviour 5 (3) (2021) 337–348
