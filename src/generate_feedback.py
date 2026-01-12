import pandas as pd
import random

DATA_PATH = "data/raw.csv"
OUTPUT_PATH = "data/with_text.csv"

negative_phrases = [
    "I feel stuck in my role",
    "No growth opportunities",
    "Salary is too low",
    "Work is stressful",
    "No recognition from management",
    "Poor work life balance",
    "No career progression",
]

positive_phrases = [
    "I love my team",
    "Good work life balance",
    "Great learning opportunities",
    "Supportive management",
    "Happy with my salary",
    "Enjoy my work",
    "Good career growth",
]


def main():
    df = pd.read_csv(DATA_PATH)

    feedback = []
    for _, row in df.iterrows():
        if row["Attrition"] == "Yes":
            feedback.append(random.choice(negative_phrases))
        else:
            feedback.append(random.choice(positive_phrases))

    df["EmployeeFeedback"] = feedback
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved dataset with text to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
