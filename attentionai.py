import streamlit as st
import sqlite3
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Define the database file
DB_FILE = "preferences.db"

# Database setup functions
def create_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS preferences (
                        user_id TEXT PRIMARY KEY,
                        city TEXT,
                        available_time TEXT,
                        budget TEXT,
                        interests TEXT,
                        starting_point TEXT
                    )''')
    conn.commit()
    conn.close()

def get_preferences(user_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM preferences WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def save_preferences(user_id, city, available_time, budget, interests, starting_point):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''REPLACE INTO preferences (user_id, city, available_time, budget, interests, starting_point)
                      VALUES (?, ?, ?, ?, ?, ?)''', (user_id, city, available_time, budget, interests, starting_point))
    conn.commit()
    conn.close()

# Load the BlenderBot model and tokenizer
def load_blenderbot_model():
    model_name = "facebook/blenderbot-400M-distill"
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Generate itinerary based on user inputs
def generate_itinerary(city, available_time, budget, interests, starting_point, model, tokenizer):
    # Customize input message to the model
    input_text = f"I am visiting {city}. I have {available_time} available and a budget of {budget}. I'm interested in {interests} and starting from {starting_point}. Can you suggest a personalized itinerary for my trip?"
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

# Streamlit App
def run_app():
    st.title("Personalized Tour Plan Bot")
    
    create_db()  # Create the database if it doesn't exist
    
    user_id = st.text_input("Enter your user ID:", "")
    if user_id:
        previous_preferences = get_preferences(user_id)
        
        if previous_preferences:
            st.write(f"Welcome back! I remember you were interested in {previous_preferences[4]} in {previous_preferences[2]}.")
        else:
            st.write("Hello! Let's plan your trip.")

        city = st.text_input("Which city are you visiting?", "")
        available_time = st.text_input("How much time do you have for the trip (e.g., 10am - 4pm)?", "")
        budget = st.text_input("What is your budget for the day?", "")
        interests = st.text_input("What are your interests? (culture, adventure, food, shopping, etc.):", "")
        starting_point = st.text_input("Where will you start from (hotel, first attraction)?", "")
        
        if st.button("Save Preferences and Get Itinerary"):
            if city and available_time and budget and interests and starting_point:
                save_preferences(user_id, city, available_time, budget, interests, starting_point)
                
                model, tokenizer = load_blenderbot_model()
                itinerary = generate_itinerary(city, available_time, budget, interests, starting_point, model, tokenizer)
                
                st.write("\nHere's your personalized itinerary:")
                st.write(itinerary)
            else:
                st.error("Please fill in all fields.")
    else:
        st.error("Please enter a user ID.")

if __name__ == "__main__":
    run_app()
