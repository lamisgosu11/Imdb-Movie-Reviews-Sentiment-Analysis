# IMDB-Movie-Reviews-Sentiment-Analysis with web UI for demo

This is my school project for my Machine Learning class at the University of Information Technology DS102.

## Getting Started

Sentiment analysis of IMDb comments with the Kaggle dataset (https://shorturl.at/aiqxS) is this research topic. <br />
In this assignment, I will try a variety of approaches. I will then compare those techniques and select the best one to use. 
### Prerequisites

The things you need before installing the repository.

* Python
* Imported libraries (Gradio for the old UI with low performance)
* For the UI, use Streamlit for better performance.
* The dataset (https://shorturl.at/aiqxS)

### Installation

A step-by-step guide will tell you how to get the development environment up and running.

```
$ You can get the dataset here: https://shorturl.at/aiqxS 
$ Set up Python and the Jupiter notebook.
$ Install the libraries that I imported in requirements.txt.
```

## Usage

A few examples of useful commands and/or tasks.

```
$ Import the dataset and library in the folder in the order of the repository.
$ Execute the .ipynb files (modify the dataset folder as necessary)
$ Examine the results.
$ We will extend these features later for future implementation (input new comments and auto sentiment identify them).
```

## Deployment

Further information on deploying this on a live or release system is available (web UI). Discusses the most important branches, the pipelines they activate, and how to keep the database up to date (if anything special). <br />
There is an issue with phrases when the word weight is too high, causing incorrect classification for sentences that are too short.
```
$ Example: I don't love this movie would be classified as positive since "Love" has a lot larger weight than "Don't," so "Don't" will not make it negative.
```
## Additional Documentation and Acknowledgments

* Credit for the dataset: Kaggle user: LAKSHMIPATHI N (https://shorturl.at/aiqxS)

## References

* Maas AL, Daly RE, Pham PT, Huang D, Ng AY, Potts C (2011) Learning Word Vectors for Sentiment Analysis. In: Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Portland, Oregon, USA, pp 142–150
