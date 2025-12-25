import json

# 1. Load your JSON data
print("Loading task_dataset.json...")
with open('task_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Define the logic to decide Easy/Medium/Hard
def determine_complexity(text):
    # In the real paper, they used AI. Here, we use length as a proxy.
    word_count = len(text.split())
    if word_count < 50:
        return "Easy"
    elif word_count < 100:
        return "Medium"
    else:
        return "Hard"

# 3. Update the data
print("Classifying tasks...")
for task in data:
    # Get the description
    description = task['problem_description']
    
    # Calculate complexity
    new_label = determine_complexity(description)
    
    # Update the JSON entry
    task['complexity_label'] = new_label
    print(f"Task: {task['task_title']} -> Classified as: {new_label}")

# 4. Save the updated JSON (Overwriting the old one)
with open('task_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("--------------------------------------------------")
print("SUCCESS: Your JSON file now has complexity labels!")