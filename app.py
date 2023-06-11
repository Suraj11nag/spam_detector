import streamlit as st
import pickle

# Load the pickled model
with open('spam_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pickled vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Define the preprocess_message function
def preprocess_message(text):
    # Lowercase the text
    text = text.lower()
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

# Create the Streamlit user interface
st.title("Spam Filter")

message = st.text_area("Enter a message")
if st.button("Predict"):
    # Preprocess the input message
    preprocessed_message = preprocess_message(message)

    # Vectorize the preprocessed message
    vectorized_message = vectorizer.transform([preprocessed_message])

    # Make the prediction
    prediction = model.predict(vectorized_message)[0]

    # Display the prediction
    if prediction == 'ham':
        st.success("Not spam")
    else:
        st.error("Spam")
