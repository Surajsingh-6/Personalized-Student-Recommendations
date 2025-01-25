import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def generate_insights_and_recommendations(topic_performance):
    
    weak_topics, strong_topics, moderate_topics = [], [], []
    for _, row in topic_performance.iterrows():
        accuracy = row['accuracy'] * 100
        print(f"{row['topic']} → Accuracy: {accuracy:.2f}%, Avg Score: {row['score']:.2f}")
        
        if row['accuracy'] < 0.5:
            print("Weak Area: Needs improvement in this topic.")
            weak_topics.append(row['topic'])
        elif row['accuracy'] >= 0.8:
            print("Strong Area: Performing well in this topic.")
            strong_topics.append(row['topic'])
        else:
            print("Moderate Area: Can improve further with more practice.")
            moderate_topics.append(row['topic'])
    
    return weak_topics, strong_topics, moderate_topics

def recommend_focus_areas(weak_topics, strong_topics, moderate_topics):
    
    print("\nRecommended Topics for Improvement:")
    for topic in weak_topics:
        print(f"Focus More On: {topic}")
    
    print("\nStrong Topics (Keep Practicing!):")
    for topic in strong_topics:
        print(f"Well Done! {topic}")
    
    print("\nModerate Topics (Can Improve Further):")
    for topic in moderate_topics:
        print(f"Keep Working On: {topic}")

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

def visualize_results(topic_performance, merged_df):
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='topic', y='accuracy', data=topic_performance, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title("Topic-Wise Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Topics")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.countplot(x='student_persona', data=merged_df, palette='viridis')
    plt.title("Distribution of Student Personas")
    plt.ylabel("Count")
    plt.xlabel("Student Personas")
    plt.show()

def main(api_url):
    data = fetch_data(api_url)
    if data is not None:
        merged_df = prepare_data(data)
        topic_performance = analyze_performance(merged_df)
        weak_topics, strong_topics, moderate_topics = generate_insights_and_recommendations(topic_performance)
        recommend_focus_areas(weak_topics, strong_topics, moderate_topics)
        merged_df = cluster_student_personas(merged_df)
        persona_summary = merged_df.groupby('student_persona').size().reset_index(name='count')
        print("\nStudent Persona Summary:\n", persona_summary)
        visualize_results(topic_performance, merged_df)

api_url = "https://api.jsonserve.com/XgAgFJ"
main(api_url)
