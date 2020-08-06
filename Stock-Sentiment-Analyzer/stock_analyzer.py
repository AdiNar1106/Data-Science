''' Analyzing stock sentiment on financial news headlines from FINVIZ.com'''
__author__ = "Aditya Narayanan"

# Web scraped files in datasets folder
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt  
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Importing files
def analyzeStocks():
    html_tables = {}

# For every table in the datasets folder...
    for table_name in os.listdir('datasets'):
        #this is the path to the file.
        table_path = f'datasets/{table_name}'
        # Open as a python file in read-only mode
        table_file = open(table_path, 'r')
        # Read the contents of the file into 'html'
        html = BeautifulSoup(table_file)
        
        html_table = html.find(id="news-table")
        # Adding the table to our dictionary
        html_tables[table_name] = html_table


    # Read one single day of headlines 
    tsla = html_tables['tsla_22sep.html']
    # Get all the table rows tagged in HTML with <tr> into 'tesla_tr'
    tsla_tr = tsla_tr = tsla.findAll('tr')

    # For each row...
    for i, table_row in enumerate(tsla_tr):
        
        link_text = table_row.a.get_text()
        
        data_text = table_row.td.get_text()
        # Print the count
        print(f'{i}:')
        # Print the contents of 'link_text' and 'data_text' 
        print(link_text)
        print(data_text)
        # The following exits the loop after three rows to prevent spamming the notebook.
        if i == 3:
            break
    
    # Hold the parsed news into a list
    parsed_news = []
    # Iterate through the news
    for file_name, news_table in html_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            
            text = x.get_text() 
            headline = x.a.get_text()
            
            date_scrape = x.td.text.split()
            
            if  len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, headline])

    
    # New words and values
    new_words = {
        'crushes': 10,
        'beats': 5,
        'misses': -5,
        'trouble': -10,
        'falls': -100,
    }
    # Instantiate the sentiment intensity analyzer with the existing lexicon
    vader = SentimentIntensityAnalyzer()
    # Update the lexicon
    vader.lexicon.update(new_words)

    columns = ['ticker', 'date', 'time', 'headline']

    scored_news = pd.DataFrame(parsed_news, columns=columns)
    # Iterate through the headlines and get the polarity scores
    scores = [vader.polarity_scores(headline) for headline in scored_news.headline.values]

    scores_df = pd.DataFrame(scores)
    # Join the DataFrames
    scored_news = pd.concat([scored_news, scores_df], axis=1)
    # Convert the date column from string to datetime
    scored_news['date'] = pd.to_datetime(scored_news.date).dt.date

    plt.style.use("fivethirtyeight")

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_c = scored_news.groupby(['date', 'ticker']).mean()

    mean_c = mean_c.unstack(level=1)

    mean_c = mean_c.xs('compound', axis=1)

    mean_c.plot.bar()
    plt.savefig("plot1.png")

    # Analyzing just one day of stock trends

    # Set the index to ticker and date
    scored_news_clean = scored_news.drop_duplicates(subset=['ticker', 'headline'])
    single_day = scored_news_clean.set_index(['ticker', 'date'])

    single_day = single_day.loc['fb']
    # Selecting the 3rd of January of 2019
    single_day = single_day.loc['2019-01-03']
    # Convert the datetime string to just the time
    single_day['time'] = pd.to_datetime(single_day['time'])
    single_day['time'] = single_day.time.dt.time 

    single_day = single_day.set_index('time')
    # Sort it
    single_day = single_day.sort_index(ascending=True)

    # Visualizing sentiment for that day
    TITLE = "Negative, neutral, and positive sentiment for FB on 2019-01-03"
    COLORS = ["red","orange", "green"]
    # Drop the columns that aren't useful for the plot
    plot_day = single_day.drop(['headline', 'compound'], axis=1)
    # Change the column names to 'negative', 'positive', and 'neutral'
    plot_day.columns = ["negative", "positive", "neutral"]

    plot_day.plot(kind='bar').legend(bbox_to_anchor=(1, 1))
    plt.savefig("plot2.png")

# Calling function 
analyzeStocks()