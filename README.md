# Alcohol Consumption in Russia

![Alcoholic Beverages in Russia](images/drinks.png?raw=true)

Source: [The Russian alcohol market: a heady cocktail](http://www.food-exhibitions.com/Market-Insights/Russia/The-Russian-alcohol-market)

This project will analyze data on alcoholic sales in Russia and develop a recommender system to recommend regions with sales similar to Saint Petersburg.

## Background

A fictitious company based in Russia owns a chain of stores that sell a variety of alcohol. A recent wine promotion in Saint Petersburg has incentivized the company to run the same across other regions. Due to cost considerations, management has decided to limit the promotion to only ten regions with similar buying habits to Saint Peterburg with the expectation of similar success in sales.

![Regions in Russia](images/regions.png?raw=true)

Source: [Outline of Russia](https://en.wikipedia.org/wiki/Outline_of_Russia)

## Method

We use Collaborative Filtering algorithm to develop the recommender system. We followed the below steps to perform the analysis:

- Exploratory Data Analysis to identify patterns.
- Implement a Collaborative Filtering algorithm to make recommendations.

## Data

The data used in this project is obtained from [Datacamp's Career Hub repository](https://github.com/datacamp/careerhub-data) on GitHub. It contains 7 variables as seen in the description below:

![Description of dataset](images/data_description.png?raw=true)

## Example plots from Exploratory Data Analysis

![Time series of alcohol sales](images/bevs_ts.png?raw=true)

The time series plot above indicates that beer had the highest sales year over year even though sales decreased from 2012 to 2015. On the other hand, our product of interest, wine, saw a gradual increase in sales starting from 2002. Vodka also experienced a gradual drop in sales. There is minimal sales increase for champagne and brandy.

Another plot worth looking at is the rank of region by alcohol sales as seen below.

![Rank of of Alcohol Sales by region](images/wine_rank.png?raw=true)

For wine-specific regional sales, we can observe that Saint Petersburg is not among the top-selling regions. This revelation could justify why management decided to embark on a wine promotion in Saint Petersburg.

## Recommender System

The below images are outputs for regions with wine sales similar to Saint Petersburg, respectively.

![Wine Sales in Saint Petersburg](images/wine_recommender.png?raw=true)

## Dependencies

- Numpy 1.19.2
- Matplotlib 3.3.2
- Pandas 1.1.5
- Seaborn 0.11.1
- Scikit -Learn 0.23.2

## Scripts

- funcs.py
- utils.py

## Notebook

- alcohol-consumption-in-russia.ipynb

## Environment

- python 3.6.12
