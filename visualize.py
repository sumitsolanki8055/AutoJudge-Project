import json
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the labeled data
with open('task_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Convert to DataFrame for easy counting
df = pd.DataFrame(data)

# 3. Count the complexity levels
counts = df['complexity_label'].value_counts()
print("Dataset Distribution:")
print(counts)

# 4. Create a Bar Chart
plt.figure(figsize=(8, 5))
colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in counts.index])

plt.title('Task Complexity Distribution')
plt.xlabel('Complexity Level')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
print("Opening visualization...")
plt.show()