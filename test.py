import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Initial data
data = {
    'Ticket Text': [
        "Server is down, need immediate assistance",
        "How do I reset my password?",
        "Payment failed multiple times, urgent fix needed",
        "Unable to login after recent update",
        "Requesting information on billing cycle",
        "Mobile app crashes occasionally",
        "Client presentation is not loading, urgent help",
        "How to access archived messages?",
        "System running very slow since last patch",
        "Request for new user account creation",
        "transaction failure"
    ],
    'Urgency': [
        
        "High", "Low", "High", "High", "Low",
        "Low", "High", "Low", "High", "Low",
        "High"
    ]
}

# 2. Create DataFrame
df = pd.DataFrame(data)

# 3. Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Ticket Text'])
y = df['Urgency']
model = LogisticRegression()
model.fit(X, y)

# 4. Loop to keep accepting inputs
while True:
    user_input = input("\nEnter your support ticket text (or type 'exit' to quit): ")
    if user_input.strip().lower() == 'exit':
        break

    # Transform input
    input_vector = vectorizer.transform([user_input])

    # Predict urgency
    prediction = model.predict(input_vector)[0]
    print(f"Predicted Urgency Level: {prediction}")

    # Ask user if the prediction was correct
    confirm = input("Is this correct? (yes/no): ").strip().lower()
    if confirm == "yes":
        true_urgency = prediction
    else:
        # Ask user for the correct urgency
        while True:
            correction = input("Please enter correct urgency (High/Low): ").strip().capitalize()
            if correction in ["High", "Low"]:
                true_urgency = correction
                break
            else:
                print("Invalid input. Please type 'High' or 'Low'.")

    # Add user input and actual urgency to the DataFrame
    new_row = {'Ticket Text': user_input, 'Urgency': true_urgency}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print("\n✅ Entry saved to dataset.")
    print(df.tail(1))  # show only the newly added entry

# Optional: Save updated data to CSV
df.to_csv("updated_tickets.csv", index=False)
print("\nAll inputs saved to 'updated_tickets.csv'. Goodbye!")

