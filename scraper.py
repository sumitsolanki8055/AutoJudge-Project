import requests
from bs4 import BeautifulSoup
import json
import time
import pandas as pd

# This list mimics the "Source Identification" step from the paper
# We will scrape the first 5 problems from Project Euler as a demo
urls_to_scrape = [
    "https://projecteuler.net/problem=1",
    "https://projecteuler.net/problem=2",
    "https://projecteuler.net/problem=3",
    "https://projecteuler.net/problem=4",
    "https://projecteuler.net/problem=5"
]

dataset = []

print(f"Starting scraping of {len(urls_to_scrape)} tasks...")

for url in urls_to_scrape:
    print(f"Fetching: {url}")
    
    try:
        # 1. Get the HTML (The "Request" step)
        response = requests.get(url)
        
        if response.status_code == 200:
            # 2. Parse the HTML (The "BeautifulSoup" step)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 3. Extract specific details (The "HTML Structure Analysis" step)
            # Note: These IDs/Classes are specific to Project Euler
            
            # Extract Title
            title_tag = soup.find('h2')
            title = title_tag.text.strip() if title_tag else "No Title"
            
            # Extract Problem Content
            content_div = soup.find('div', class_='problem_content')
            description = content_div.text.strip() if content_div else "No Description"
            
            # Extract ID (from the URL)
            task_id = url.split('=')[-1]

            # Create the data entry
            task_entry = {
                "task_id": task_id,
                "task_title": title,
                "problem_description": description,
                "url": url,
                "complexity_label": "Unknown" # We would need a model to predict this later!
            }
            
            dataset.append(task_entry)
            print(f" -> Successfully scraped: {title}")
            
            # Sleep to be polite to the server
            time.sleep(1)
            
        else:
            print(f" -> Failed to retrieve {url} (Status: {response.status_code})")

    except Exception as e:
        print(f" -> Error: {e}")

# 4. Save to JSON and CSV (The "Final Output" step)
print("--------------------------------------------------")
print("Saving data...")

# Save as JSON (like in the paper)
with open('task_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=4)

# Also save as CSV (easier to view in Excel/VS Code)
df = pd.DataFrame(dataset)
df.to_csv('task_dataset.csv', index=False)

print(f"DONE! Scraped {len(dataset)} tasks.")
print("Check 'task_dataset.json' and 'task_dataset.csv' in your folder.")