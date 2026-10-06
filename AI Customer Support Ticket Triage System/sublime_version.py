# AI Customer Support Ticket Triage System - Sublime Text Edition
# Just run this in Sublime: Tools > Build > or Ctrl+B

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🎫 TICKET TRIAGE SYSTEM")
print("="*70 + "\n")

# ============================================================================
# Generate Training Data
# ============================================================================

def generate_data():
    np.random.seed(42)

    templates = {
        'billing': [
            "I was charged twice for my subscription",
            "Why is my invoice $99 instead of $29?",
            "Can I get a refund for the duplicate charge?",
            "My payment failed but I was still charged",
        ],
        'technical': [
            "The app keeps crashing",
            "I'm getting a 500 error",
            "The API endpoint is timing out",
            "My integration stopped working",
        ],
        'account': [
            "I forgot my password",
            "How do I enable two-factor authentication?",
            "I need to change my email",
            "Can you recover my deleted account?",
        ],
        'product': [
            "This feature isn't working",
            "I'd like to request a new feature",
            "The latest update broke my workflow",
            "How do I use this new feature?",
        ]
    }

    tickets = []
    for _ in range(500):
        category = np.random.choice(list(templates.keys()))
        text = np.random.choice(templates[category])
        urgency = 'high' if np.random.random() < 0.5 else 'low'
        tickets.append({'text': text, 'category': category, 'urgency': urgency})

    return pd.DataFrame(tickets)

# ============================================================================
# Train Models
# ============================================================================

print("📊 Generating data...")
df = generate_data()
print(f"✓ Generated {len(df)} tickets\n")

print("🔄 Training models...")
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = tfidf.fit_transform(df['text'].str.lower())

cat_enc = LabelEncoder()
urg_enc = LabelEncoder()
y_cat = cat_enc.fit_transform(df['category'])
y_urg = urg_enc.fit_transform(df['urgency'])

X_train, X_test, yc_train, yc_test, yu_train, yu_test = train_test_split(
    X, y_cat, y_urg, test_size=0.2, random_state=42
)

cat_model = OneVsRestClassifier(LinearSVC(random_state=42, max_iter=2000))
urg_model = RandomForestClassifier(n_estimators=100, random_state=42)

cat_model.fit(X_train, yc_train)
urg_model.fit(X_train, yu_train)
print("✓ Models trained!\n")

# ============================================================================
# Evaluate
# ============================================================================

print("="*70)
print("RESULTS")
print("="*70 + "\n")

yc_pred = cat_model.predict(X_test)
yu_pred = urg_model.predict(X_test)

print(f"Category F1-Score: {f1_score(yc_test, yc_pred, average='weighted'):.4f}")
print(f"Urgency F1-Score: {f1_score(yu_test, yu_pred, average='weighted'):.4f}\n")

print("Category Classification Report:")
print(classification_report(yc_test, yc_pred, target_names=cat_enc.classes__))

# ============================================================================
# Make Predictions
# ============================================================================

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(text):
    X = tfidf.transform([text.lower()])

    c_idx = cat_model.predict(X)[0]
    c_scores = cat_model.decision_function(X)[0]
    c_conf = float(softmax(c_scores)[c_idx])

    u_idx = urg_model.predict(X)[0]
    u_proba = urg_model.predict_proba(X)[0]
    u_conf = float(u_proba[u_idx])

    return {
        'category': cat_enc.inverse_transform([c_idx])[0],
        'cat_conf': c_conf,
        'urgency': urg_enc.inverse_transform([u_idx])[0],
        'urg_conf': u_conf,
    }

print("\n" + "="*70)
print("PREDICTIONS")
print("="*70 + "\n")

tests = [
    "I was charged twice, refund needed immediately!",
    "API is down affecting all production users",
    "Forgot my password need to reset",
    "How do I use the new dashboard?",
]

for i, ticket in enumerate(tests, 1):
    result = predict(ticket)
    print(f"{i}. {ticket[:45]}...")
    print(f"   Category: {result['category']} ({result['cat_conf']:.0%})")
    print(f"   Urgency: {result['urgency']} ({result['urg_conf']:.0%})\n")

print("="*70)
print("✅ SYSTEM COMPLETE!")
print("="*70)