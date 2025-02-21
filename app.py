import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Function to render Home page
def show_home():
    st.title("IPL Win Predictor")
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the batting team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the bowling team', sorted(teams))
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams must be different!")
        st.stop()
    selected_city = st.selectbox('Select host city', sorted(cities))
    target = st.number_input('Target', min_value=0, format="%d")
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score', min_value=0, format="%d")
    with col4:
        overs = st.number_input(
            'Overs completed',
            min_value=0.0,
            max_value=20.0,
            step=0.1,
            format="%.1f"
        )
    with col5:
        wickets_out = st.number_input('Wickets out', min_value=0, format="%d")
    if st.button('Predict Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(f'{batting_team} - {round(win * 100)}%')
        st.header(f'{bowling_team} - {round(loss * 100)}%')
        return batting_team, round(win * 100), bowling_team, round(loss * 100)
    return None, None, None, None

# Function to render About Us page
def show_about():
    st.title("About Us")
    st.write("""  
    ### IPL Win Predictor  
    This application is designed to predict the probability of winning an IPL match based on real-time data.  
    Using advanced machine learning techniques, we analyze match conditions and provide insightful predictions.  

    **Key Features:**  
    - Machine learning-driven probability predictions.  
    - User-friendly interface for cricket enthusiasts.  
    - Updated model with hyperparameter tuning for better accuracy.  

    **Datasets:**
    The project utilizes two primary datasets:
    1. **Matches Dataset (`matches.csv`)**: Contains detailed information about IPL matches, including teams, scores, outcomes, player of the match, venue, and more.
    2. **Deliveries Dataset (`deliveries.csv`)**: Contains ball-by-ball data for each match, providing granular details like batsman, bowler, runs scored, extras, dismissals, etc.

    **Two machine learning algorithms were employed:**
    1. **Logistic Regression**:
    - A statistical model that uses a logistic function to model a binary dependent variable.
    - Suitable for binary classification tasks like predicting win or loss.

    2. **Random Forest Classifier**:
    - An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.
    - Handles overfitting better and captures complex patterns in the data.

    3. **Hyperparameter Tuning Algorithm**:

        1. **Grid Search:**
            - Exhaustively searches predefined hyperparameter values to find the best combination.

        2. **Random Search:**
            - Randomly selects hyperparameter values within a range, balancing efficiency and performance.
            - Both models were trained and evaluated to compare performance, with hyperparameter tuning applied to optimize their predictive capabilities.

    **Ensure the following Python libraries are installed:**
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `matplotlib`
    - `seaborn`
    - `jupyter`

    **Developed By:**  
    - Parth Doshi  
    - Contact: pmdoshi21@gmail.com  
    """)

# Initialize navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2 = st.columns([1, 8], gap="small")

with col1:
    if st.button("Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("About Us"):
        st.session_state.page = "About Us"

# selected page
if st.session_state.page == "Home":
    batting_team, win_percentage, bowling_team, loss_percentage = show_home()
    if batting_team and win_percentage and bowling_team and loss_percentage:
        # Data for the bar chart
        teams = [batting_team, bowling_team]
        probabilities = [win_percentage, loss_percentage]
        # Create the bar chart
        fig, ax = plt.subplots()
        bars = ax.bar(teams, probabilities, color=['green', 'red'])
        # Add labels and title
        ax.set_ylabel('Probability (%)')
        ax.set_title('IPL Win Probability')
        # Display percentage on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval}%', ha='center', va='bottom')
        # Display the chart in Streamlit
        st.pyplot(fig)
elif st.session_state.page == "About Us":
    show_about()
