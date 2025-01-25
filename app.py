from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

def fetch_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(data):
    
    df = pd.DataFrame(data)
    df2 = df['quiz'].apply(pd.Series)
    df2 = df2[['id', 'title', 'difficulty_level', 'questions_count', 'topic', 'duration']]
    df = df[['quiz_id', 'user_id', 'score', 'trophy_level', 'accuracy', 'speed', 
             'final_score', 'negative_score', 'correct_answers', 'incorrect_answers', 
             'duration', 'better_than', 'rank_text', 'mistakes_corrected', 'initial_mistake_count']]
    df['accuracy'] = df['accuracy'].str.replace('%', '').astype(float) / 100  
    df['score'] = pd.to_numeric(df['score'], errors='coerce') 
    df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')  

    merged_df = df.merge(df2, left_on='quiz_id', right_on='id', how='left')
    return merged_df

def analyze_performance(merged_df):
    
    topic_performance = merged_df.groupby('topic').agg({
        'accuracy': 'mean',
        'score': 'mean',
        'final_score': 'mean',
        'incorrect_answers': 'sum',
        'correct_answers': 'sum'
    }).reset_index()
    return topic_performance

#reccomendation system
def generate_insights_and_recommendations(topic_performance):
    weak_topics, strong_topics, moderate_topics = [], [], []
    
    for index, row in topic_performance.iterrows():
        if row['accuracy'] < 0.5:
            weak_topics.append(row['topic'])
        elif row['accuracy'] >= 0.8:
            strong_topics.append(row['topic'])
        else:
            moderate_topics.append(row['topic'])
    
    return weak_topics, strong_topics, moderate_topics

# AI/ML Function to cluster student personas
def cluster_student_personas(merged_df):
    
    clustering_features = merged_df[['accuracy', 'score', 'final_score', 
                                      'incorrect_answers', 'correct_answers']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features.fillna(0))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    merged_df['persona_cluster'] = kmeans.fit_predict(scaled_features)
    persona_labels = {
        0: "Beginner - Needs Improvement",
        1: "Intermediate - Consistent Performer",
        2: "Advanced - High Achiever"
    }
    merged_df['student_persona'] = merged_df['persona_cluster'].map(persona_labels)
    return merged_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        api_url = request.form['api_url']
        data = fetch_data(api_url)
        if data is not None:
            merged_df = prepare_data(data)
            topic_performance = analyze_performance(merged_df)
            weak_topics, strong_topics, moderate_topics = generate_insights_and_recommendations(topic_performance)
            merged_df = cluster_student_personas(merged_df)
            persona_summary = merged_df.groupby('student_persona').size().reset_index(name='count').to_dict(orient='records')
            return render_template('results.html',
                                   topic_performance=topic_performance.to_dict(orient='records'),
                                   weak_topics=weak_topics,
                                   strong_topics=strong_topics,
                                   moderate_topics=moderate_topics,
                                   persona_summary=persona_summary)     
        return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
