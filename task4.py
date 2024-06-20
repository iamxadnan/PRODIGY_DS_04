import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

twitter_training = pd.read_csv('twitter_training.csv', names=['ID', 'Topic', 'Sentiment', 'Tweet'])
twitter_validation = pd.read_csv('twitter_validation.csv', names=['ID', 'Topic', 'Sentiment', 'Tweet'])

data = pd.concat([twitter_training, twitter_validation], ignore_index=True)

print(data.head())

sentiment_counts = data.groupby(['Topic', 'Sentiment']).size().reset_index(name='Counts')

plt.figure(figsize=(15, 10))
sns.barplot(x='Topic', y='Counts', hue='Sentiment', data=sentiment_counts)
plt.title('Sentiment Distribution by Topic')
plt.xticks(rotation=90)
plt.ylabel('Number of Tweets')
plt.xlabel('Topic')
plt.show()

plt.figure(figsize=(10, 10))
pie_data = data['Sentiment'].value_counts()
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Overall Sentiment Distribution')
plt.show()

topic = 'Borderlands'
plt.figure(figsize=(10, 10))
topic_data = data[data['Topic'] == topic]
topic_pie_data = topic_data['Sentiment'].value_counts()
plt.pie(topic_pie_data, labels=topic_pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title(f'Sentiment Distribution for {topic}')
plt.show()

positive_tweets = data[data['Sentiment'] == 'Positive']['Tweet'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Sentiments')
plt.show()

fig = px.sunburst(data, path=['Topic', 'Sentiment'], values='ID', title='Sunburst Chart of Sentiments by Topic')
fig.show()

fig = go.Figure()
for sentiment in data['Sentiment'].unique():
    sentiment_data = sentiment_counts[sentiment_counts['Sentiment'] == sentiment]
    fig.add_trace(go.Bar(x=sentiment_data['Topic'], y=sentiment_data['Counts'], name=sentiment))

fig.update_layout(
    title='Interactive Sentiment Distribution by Topic',
    xaxis_title='Topic',
    yaxis_title='Number of Tweets',
    barmode='stack'
)
fig.show()
