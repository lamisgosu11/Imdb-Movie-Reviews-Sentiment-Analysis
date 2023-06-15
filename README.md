# Imdb-Movie-Reviews-Sentiment-Analysis with web UI for demo

This is my school project for my Machine Learning class at the University of Information Technology (DS103.N21).

## Getting Started

Sentiment analysis of imdb comments with kaggle dataset (https://shorturl.at/aiqxS) is the topic of this research. <br />
In this assignment, I will try a variety of approaches. I will then compare those techniques and select the best one to use. 
### Prerequisites

The things you need before installing the repository.

* Python
* Imported libraries (gradio for UI)
* The dataset (https://shorturl.at/aiqxS)

### Installation

A step by step guide that will tell you how to get the development environment up and running.

```
$ You can get the dataset here: https://shorturl.at/aiqxS 
$ Set up Python and the Jupiter notebook.
$ Install the libraries that I imported.
```

## Usage

A few examples of useful commands and/or tasks.

```
$ Import dataset and library in the folder like the order of the repository.
$ Execute the.ipynb files (modify the dataset folder as necessary)
$ Examine the results.
$ We will extend these features later for future implementation (input new comments and auto sentiment identify them).
```

## Deployment

Further information on how to deploy this on a live or release system is available (web ui). Discusses the most important branches, the pipelines they activate, and how to keep the database up to date (if anything special). <br />
There is an issue with phrases when the word weight is too high, causing incorrect classification for sentences that are too short.
```
$ Example: I don't love this movie would be classified as positive since "Love" has a lot larger weight than "Don't," so "Don't" will not make it negative.
```
## Additional Documentation and Acknowledgments

* Project folder on server:
* Credit for the dataset: Kaggle user: LAKSHMIPATHI N (https://shorturl.at/aiqxS)

## References

* Maas AL, Daly RE, Pham PT, Huang D, Ng AY, Potts C (2011) Learning Word Vectors for Sentiment Analysis. In: Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Portland, Oregon, USA, pp 142–150
