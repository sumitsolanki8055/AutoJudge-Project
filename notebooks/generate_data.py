import pandas as pd
import random

# Define keywords to simulate difficulty (simulating the "Ground Truth")
topics = {
    "Easy": ["print", "loop", "if statement", "array", "sum", "average", "basic math", "convert", "count"],
    "Medium": ["binary search", "sorting", "stack", "queue", "hash map", "two pointers", "greedy", "recursion"],
    "Hard": ["dynamic programming", "graph", "tree", "shortest path", "segment tree", "bitmask", "network flow"]
}

data = []

print("Generating synthetic training data...")

# Generate 500 synthetic problems
for i in range(500):
    # 1. Randomly pick a difficulty
    difficulty = random.choice(["Easy", "Medium", "Hard"])
    
    # 2. Assign a Score based on difficulty (Regression Target)
    if difficulty == "Easy":
        score = random.randint(1, 30)
        keywords = random.sample(topics["Easy"], 2)
    elif difficulty == "Medium":
        score = random.randint(31, 70)
        keywords = random.sample(topics["Medium"], 2)
    else:
        score = random.randint(71, 100)
        keywords = random.sample(topics["Hard"], 2)

    # 3. Create Text Fields (Features) matching the PDF requirements
    title = f"Problem {i+1}: {keywords[0].capitalize()} Task"
    desc = f"Write a program that uses {keywords[0]} and {keywords[1]} to solve the problem. The constraints are small."
    inp = "The first line contains an integer N."
    out = "Print the result."
    
    # Add noise to make it realistic
    if random.random() > 0.5:
        desc += " This is a standard optimization problem."

    data.append([title, desc, inp, out, difficulty, score])

# Save to CSV
df = pd.DataFrame(data, columns=["title", "description", "input_description", "output_description", "problem_class", "problem_score"])
df.to_csv("programming_problems.csv", index=False)

print("SUCCESS: 'programming_problems.csv' created with 500 samples!")
