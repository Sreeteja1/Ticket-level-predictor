from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

app = FastAPI()

# File to persist data
csv_file = "model_data.csv"

# Step 1: Load or create dataset with 3 urgency levels
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame({
        'Ticket Text': [
            "Server is down, need immediate assistance",     # High
            "How do I reset my password?",                   # Low
            "Payment failed multiple times, urgent fix",     # High
            "Unable to login after recent update",           # Medium
            "Requesting billing cycle information",          # Low
            "App crashes once a week",                       # Medium
            "Client presentation not loading, urgent",       # High
            "How to access archived messages?",              # Low
            "System very slow since last patch",             # Medium
            "Request for user account creation",             # Low
            "Transaction failure - need fix now",            # High
            "Login issue with some delay",                   # Medium
            "Report generation taking longer than usual",    # Medium
            "Want to update my contact info",                # Low
            "Getting intermittent errors in dashboard"       # Medium
        ],
        'Urgency': [
            "High", "Low", "High", "Medium", "Low",
            "Medium", "High", "Low", "Medium", "Low",
            "High", "Medium", "Medium", "Low", "Medium"
        ]
    })

# Step 2: Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Ticket Text'])
y = df['Urgency']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 3: Request Models
class Ticket(BaseModel):
    ticket: str

class Feedback(BaseModel):
    text: str
    actual_urgency: str  # High / Medium / Low

# Step 4: Prediction Endpoint
@app.post("/predict")
def predict(ticket: Ticket):
    input_vector = vectorizer.transform([ticket.ticket])
    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    class_labels = model.classes_
    
    prob_dict = {label: round(prob * 100, 2) for label, prob in zip(class_labels, probabilities)}
    
    return {
        "predicted_urgency": prediction,
        "confidence_scores": prob_dict
    }

# Step 5: Feedback Loop to Improve Accuracy
@app.post("/feedback")
def update_model(feedback: Feedback):
    global df, model, vectorizer
    new_row = {'Ticket Text': feedback.text, 'Urgency': feedback.actual_urgency}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Retrain model
    X = vectorizer.fit_transform(df['Ticket Text'])
    y = df['Urgency']
    model.fit(X, y)

    # Save updated data
    df.to_csv(csv_file, index=False)
    return {"message": "Feedback recorded and model retrained successfully."}
