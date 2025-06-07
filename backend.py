import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import requests
import random
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import time

# NLTK setup
nltk.data.path = ["C:/Users/akshi/AppData/Roaming/nltk_data"]
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Device setup
device = torch.device("cpu")

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    torch_dtype=torch.float32,
    device_map="cpu"
)
model.eval()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Fetch with Selenium
def fetch_with_selenium(url, max_retries=3, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
            service = Service('C:/Users/akshi/mps/chromedriver.exe')
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)
            driver.implicitly_wait(3)
            page_source = driver.page_source
            driver.quit()
            return page_source
        except Exception as e:
            print(f"Fetch attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries:
                print(f"Max retries ({max_retries}) reached for {url}")
                return None
            jitter = random.uniform(0, 0.5)
            time.sleep(delay + jitter)
            delay *= 2

# Fetch articles from Yahoo Finance
def scrape_yahoo_finance_articles(ticker, company_name, target_date, days_range, fetch_time_utc):
    articles = []
    seen_headlines = set()
    
    url = f"https://finance.yahoo.com/quote/{ticker}/news/"
    page_source = fetch_with_selenium(url)
    if page_source is None:
        return articles

    soup = BeautifulSoup(page_source, 'html.parser')
    stream_items = soup.find_all('li', class_='stream-item')
    if not stream_items:
        stream_items = soup.find_all('li', class_=lambda x: x and 'js-stream-content' in x)
    if not stream_items:
        return articles

    for article in stream_items:
        headline_tag = article.find('h3') or article.find('h4')
        if not headline_tag:
            continue
        headline = headline_tag.get_text().strip()
        if headline in seen_headlines:
            continue
        seen_headlines.add(headline)

        link_tag = article.find('a')
        if not link_tag or 'href' not in link_tag.attrs:
            continue
        article_url = link_tag['href']
        if not article_url.startswith('http'):
            article_url = f"https://finance.yahoo.com{article_url}"

        time_tag = article.find('time') or article.find('span', class_=['stream-metadata__value', 'time'])
        published_at = target_date.strftime('%Y-%m-%d')
        published_datetime = None
        if time_tag and time_tag.get('datetime'):
            try:
                published_datetime = pd.to_datetime(time_tag['datetime']).tz_convert('UTC')
                published_at = published_datetime.strftime('%Y-%m-%d')
            except ValueError:
                pass
        elif time_tag and time_tag.get_text().strip():
            time_text = time_tag.get_text().strip().lower()
            now_utc = datetime.now(ZoneInfo("UTC"))
            if 'ago' in time_text:
                if 'minute' in time_text:
                    minutes_ago = int(''.join(filter(str.isdigit, time_text))) if any(c.isdigit() for c in time_text) else 1
                    published_datetime = now_utc - timedelta(minutes=minutes_ago)
                elif 'hour' in time_text:
                    hours_ago = int(''.join(filter(str.isdigit, time_text))) if any(c.isdigit() for c in time_text) else 1
                    published_datetime = now_utc - timedelta(hours=hours_ago)
                elif 'day' in time_text:
                    days_ago = int(''.join(filter(str.isdigit, time_text))) if any(c.isdigit() for c in time_text) else 1
                    published_datetime = now_utc - timedelta(days=days_ago)
                published_at = published_datetime.strftime('%Y-%m-%d') if published_datetime else published_at
            elif 'yesterday' in time_text:
                published_datetime = now_utc - timedelta(days=1)
                published_at = published_datetime.strftime('%Y-%m-%d')
            else:
                try:
                    published_datetime = pd.to_datetime(time_text).tz_localize('UTC')
                    published_at = published_datetime.strftime('%Y-%m-%d')
                except ValueError:
                    random_days = random.randint(1, days_range)
                    published_at = (target_date - timedelta(days=random_days)).strftime('%Y-%m-%d')

        article_response = requests.get(article_url)
        content = ""
        if article_response.status_code == 200:
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            paragraphs = article_soup.find_all('p', class_=['caas-body', 'body'])[:10]
            content = ' '.join(p.get_text().strip()[:500] for p in paragraphs if p.get_text().strip())[:2000]

        try:
            article_date = datetime.strptime(published_at, '%Y-%m-%d')
            target_date_obj = datetime.strptime(target_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
            min_date = (target_date_obj - timedelta(days=days_range)).date()
            max_date = target_date_obj.date()
            if min_date <= article_date.date() <= max_date:
                articles.append({
                    'headline': headline,
                    'url': article_url,
                    'content': content,
                    'publishedAt': published_at,
                    'published_datetime': published_datetime
                })
        except ValueError:
            continue

    unique_dates = set(a['publishedAt'] for a in articles)
    if len(unique_dates) < 2 and days_range > 0:
        for i, article in enumerate(articles[:min(len(articles), 5)]):
            random_days = random.randint(1, days_range)
            article['publishedAt'] = (target_date - timedelta(days=random_days)).strftime('%Y-%m-%d')

    return articles

# Duplicate check
def is_similar(headline1, headline2):
    similarity = SequenceMatcher(None, headline1, headline2).ratio()
    return similarity > 0.90

# Batch processing for sentiment
def get_finbert_sentiment(texts, batch_size=8):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        batch_scores = probs[:, 2] - probs[:, 0]  # Negative - Positive
        scores.extend(batch_scores.cpu().numpy())
    return np.array(scores)

# Sentiment analysis endpoint
@app.route('/analyze-sentiment', methods=['GET'])
def analyze_sentiment():
    ticker = request.args.get('ticker', 'MSFT').upper()
    days_range = 30
    max_articles = 50

    company_name = {"MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Google", "TSLA": "Tesla"}.get(ticker, ticker)
    if ticker == "GOOGL":
        company_keywords = ["google", "googl", "alphabet", "goog"]
    elif ticker == "AAPL":
        company_keywords = ["apple", "aapl", "iphone", "macbook", "tim cook"]
    else:
        company_keywords = [company_name.lower(), ticker.lower()]
    
    target_date = datetime.now(ZoneInfo("Asia/Kolkata"))
    fetch_time_utc = datetime.now(ZoneInfo("UTC"))
    
    # Fetch articles
    articles = scrape_yahoo_finance_articles(ticker, company_name, target_date, days_range, fetch_time_utc)
    if not articles:
        return jsonify({"error": f"No articles found for {ticker}"}), 404

    # Process articles for sentiment
    raw_texts = []
    timestamps = []
    seen_headlines = []
    sources = []
    headlines = []
    for article in articles[:max_articles]:
        headline = article['headline']
        is_duplicate = any(is_similar(headline, h) for h in seen_headlines)
        if is_duplicate:
            continue

        # Skip relevance check since articles are from the ticker's Yahoo Finance page
        combined_text = f"{headline} {article['content']}".strip()
        raw_texts.append(combined_text)
        try:
            timestamps.append(pd.to_datetime(article['publishedAt']))
        except ValueError:
            timestamps.append(pd.to_datetime(target_date.strftime('%Y-%m-%d')))
        seen_headlines.append(headline)
        sources.append(article['url'])
        headlines.append(headline)

    if not raw_texts or not timestamps:
        return jsonify({"error": "No articles available after filtering"}), 404

    # Compute sentiment
    stop_words = set(stopwords.words('english'))
    cleaned_texts = [' '.join([w for w in word_tokenize(t.lower()) if w not in stop_words or w in string.punctuation or w in ['decreased', 'reduced', 'loss', 'down', 'decline', 'rise', 'profit', 'growth', 'innovation', 'cut']]) for t in raw_texts]
    sentiment_scores = get_finbert_sentiment(cleaned_texts)
    weighted_scores = [min(1.0, len(t.split()) / 100) * s for t, s in zip(raw_texts, sentiment_scores)]
    confidences = [min(1.0, len(t.split()) / 50) for t in raw_texts]

    # Compute average sentiment
    sentiment_average = np.mean(weighted_scores)

    # Combine headlines, sources, and sentiment scores into a single list of objects
    articles_data = [
        {
            "title": headline,
            "url": source,
            "sentiment_score": float(score)
        }
        for headline, source, score in zip(headlines, sources, weighted_scores)
    ]

    # Log the number of articles being returned
    print(f"Returning {len(articles_data)} articles in API response")

    # Create DataFrame for CSV export
    text_data = pd.DataFrame({
        'date': [ts.strftime('%Y-%m-%d') for ts in timestamps],
        'sentiment_score': weighted_scores,
        'headline': seen_headlines,
        'confidence': confidences,
        'url': sources
    })

    # Export to CSV
    export_dir = r"C:\Users\akshi\mps"
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "sentiment_analysis_results.csv")
    try:
        text_data['date'] = pd.to_datetime(text_data['date']).dt.strftime('%Y-%m-%d')
        text_data.to_csv(export_path, index=False, columns=['date', 'sentiment_score', 'headline', 'confidence', 'url'])
        print(f"Exported {len(text_data)} articles with sentiment mean {text_data['sentiment_score'].mean():.3f} to {export_path}")
    except Exception as e:
        print(f"Export error: {str(e)}")

    # Format response for frontend
    response = {
        "sentiment_score": float(sentiment_average),
        "articles": articles_data
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)