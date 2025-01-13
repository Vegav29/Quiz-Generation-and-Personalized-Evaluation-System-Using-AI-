import json 
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
import plotly.graph_objects as go
# Set the Streamlit page configuration (This should be the first Streamlit command)
st.set_page_config(
    page_title="AI Quiz Feedback",
    page_icon="📊",
    layout="wide",  # Set layout to wide
    initial_sidebar_state="collapsed",
)

# Function to set gradient background and font color
def set_background():
    st.markdown(
        """
          <style>
        body {
            background: linear-gradient(135deg, #f8f3e8, #e9e0d1);
            color: black;
        }
        .stApp {
            background: linear-gradient(135deg, #f8f3e8, #e9e0d1);
            color: black;
        }
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: black !important;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            border: 2px solid black;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def initialize_model():
    api_key = your_api_key  # Replace with your actual API key
    return GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# Static questions for the quiz
STATIC_QUESTIONS = [
    "What is the capital of France?: Paris",
    "Explain the process of photosynthesis: Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "Describe the main themes of Shakespeare's Hamlet.: Revenge, madness, and mortality.",
    "What is the Pythagorean theorem?: In a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.",
    "Summarize the events of World War II: World War II was a global conflict that lasted from 1939 to 1945 and involved many countries.",
    "How does the water cycle work?: The water cycle is the continuous movement of water on, above, and below the surface of the Earth.",
    "Explain the theory of relativity: The theory of relativity, proposed by Albert Einstein, describes the relationship between space and time.",
    "What is the importance of the Internet?: The Internet is a global network that connects people, information, and resources.",
    "How do computers use binary code?: Computers use binary code, which consists of 0s and 1s, to represent and process data.",
    "What are the causes and effects of climate change?: Climate change is primarily caused by human activities and has far-reaching environmental impacts.",
]
def generate_individual(STATIC_QUESTIONS, llm):
    prompt_template = PromptTemplate(
        input_variables=["STATIC_QUESTIONS"],
        template="""
         You are a fun and motivating teacher who use uses lots of emojis to  tasked with evaluating a student's response to a specific question, be optimistic and motivational in your feedback
        
        questions and answers: {STATIC_QUESTIONS}
        
       You are a highly skilled assistant focused on evaluating user responses in text-based Q&A tasks. Your job is to generate a **performance table** for 10 questions, displaying scores and feedback for the following metrics: 

1. 🎯 **Relevance **: How well the response aligns with the question's context and requirements.
2. ✅ **Accuracy**: The correctness and factuality of the response.
3. 💡 **Clarity **: The simplicity, coherence, and ease of understanding in the answer.
4. 🌊 **Depth **: The level of detail, explanation, and elaboration in the response.

### **Your task:**
1. For each question, provide a **detailed performance table row** with:
   - The question text.
   - Scores (in percentages) for Relevance, Accuracy, Clarity, and Depth.
   - Fun and intuitive emoji-based tags like (🌟, ⚠️, 🗣️, 📖, etc.) to make feedback engaging.
   - Constructive feedback that highlights:
     - Strengths.
     - Areas for improvement.
     - Suggestions for better performance.



2.. Ensure your output is intuitive, fun, and motivational. Use a conversational tone, emoji, and storytelling elements where possible.

### **Output Format:**
1. **Performance Table:**
    | **Question**                  | 🎯 **Relevance (%)** | ✅ **Accuracy (%)** | 💡 **Clarity (%)** | 🌊 **Depth (%)** | **Feedback**                                                                                                                                           |
    |-------------------------------|----------------------|---------------------|--------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
    | Q1: Question text             | [Score + Emoji]     | [Score + Emoji]    | [Score + Emoji]    | [Score + Emoji] | [2 line of Feedback of their responseYour goal is to assess the quality of their responses across multiple dimensions and provide a 2 line  evaluation against the question be more specific on student Consider the following,always be motivation and optimistic in your feedback]                                                                                                     |
    | Q2: Question text             | [Score + Emoji]     | [Score + Emoji]    | [Score + Emoji]    | [Score + Emoji] | [2 line of Feedback of their response Your goal is to assess the quality of their responses across multiple dimensions and provide a 2 line evaluation against the question be more specific on student Consider the following,always be motivation and optimistic in your feedback]]                                                                                                     |
    (Repeat for all 10 questions)

### **Key Instructions:**
- Use engaging language, emojis, and intuitive formatting.
- Avoid generic feedback—make it specific to each question and metric.
- Ensure the output is insightful, fun, and valuable for the user.



        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    tab = chain.invoke({"STATIC_QUESTIONS": STATIC_QUESTIONS})
    return tab['text'].strip()
# Generate individual feedback
def generate_individual_feedback(STATIC_QUESTIONS, llm):
    prompt_template = PromptTemplate(
        input_variables=["STATIC_QUESTIONS"],
        template="""  
     You are a fun and motivating teacher. Your task is to evaluate a student's response to a question,the output should be only 10 scores and no texts

        Questions and answers: {STATIC_QUESTIONS}

        Please provide the following scores based on all 10 questions and answers:
        1. **Relevance** - a percentage score (0-100)
        2. **Accuracy** - a percentage score (0-100)
        3. **Clarity** - a percentage score (0-100)
        4. **Depth** - a percentage score (0-100)
        5. **Conceptual Understanding** - a percentage score (0-100)
        6. **Analytical Skills** - a percentage score (0-100)
        7. **Problem-Solving Ability** - a percentage score (0-100)
        8. **Critical Thinking** - a percentage score (0-100)
        9. **Application and Examples** - a percentage score (0-100)
        10. **Overall Score** as the average of all the above metrics - a percentage score (0-100)
        

        For example: 85, 90, 88, 92, 87, 91, 89, 94, 86, 89
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    feed = chain.invoke({"STATIC_QUESTIONS": STATIC_QUESTIONS})
    return feed['text'].strip()

# Parse overall scores from LLM output
def parse_scores(output):
    try: 
        # Print LLM output for debugging
        print("LLM Output Before Parsing:", output)

        # Extract numeric values only
        output = ''.join(c if c.isdigit() or c == ',' else '' for c in output)
        scores = output.split(",")
        scores = [int(score.strip()) for score in scores if score.strip().isdigit()]
        if len(scores) == 10:  # Includes the overall score
            return {
                "Relevance": scores[0],
                "Accuracy": scores[1],
                "Clarity": scores[2],
                "Depth": scores[3],
                "Conceptual Understanding": scores[4],
                "Analytical Skills": scores[5],
                "Problem-Solving Ability": scores[6],
                "Critical Thinking": scores[7],
                "Application and Examples": scores[8],
                "Overall Score": scores[9],
            }
    except Exception as e:
        st.error(f"Error parsing the scores: {e}")
        return {}

def generate_radar_plot(scores, background_color="#f8f3e8"):
    try:
        categories = ["Relevance", "Accuracy", "Clarity", "Depth"]
        values = [
            scores["Relevance"],
            scores["Accuracy"],
            scores["Clarity"],
            scores["Depth"],
        ]
        values += [values[0]]  # Close the radar chart

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name="Overall Performance",
            fillcolor="rgba(0, 128, 255, 0.5)",  # Semi-transparent blue fill
            line=dict(color="rgba(0, 255, 255, 1)", width=3)  # Cyan outline
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="rgba(0, 0, 0, 0.2)",  # Subtle gray grid lines
                    gridwidth=2,
                ),
                angularaxis=dict(
                    gridcolor="rgba(0, 0, 0, 0.2)",
                ),
            ),
            plot_bgcolor="rgba(0,0,0,0)",  # Main background color
            paper_bgcolor="rgba(0,0,0,0)",  # Main background color
            showlegend=False,
            font=dict(color="black", size=16),  # Font for labels
            title=dict(
                text="Radar Chart: Overall Performance",
                font=dict(size=20, color="black"),  # Title styling
                x=0.5,  # Center align
            ),
        )
        return fig
    except KeyError as e:
        print(f"Error generating radar plot: Missing key {e}")
        return None
def generate_bar_plot(scores, background_color="#f8f3e8"):
    try:
        selected_metrics = [
            "Depth",
            "Conceptual Understanding",
            "Analytical Skills",
            "Problem-Solving Ability",
            "Critical Thinking",
        ]
        selected_values = [scores[metric] for metric in selected_metrics]

        fig = go.Figure([go.Bar(x=selected_metrics, y=selected_values, marker_color='lightblue')])

        fig.update_layout(
            title=dict(
                text="Performance Metrics (4-8)",
                font=dict(size=20, color="black"),  # Title styling with black font
                x=0.5,  # Center alignment
            ),
            xaxis=dict(
                title=dict(text="Metrics", font=dict(size=16, color="black")),  # Black font for x-axis title
                tickfont=dict(color="black", size=14),  # Black font for x-axis tick labels
            ),
            yaxis=dict(
                title=dict(text="Scores", font=dict(size=16, color="black")),  # Black font for y-axis title
                tickfont=dict(color="black", size=14),  # Black font for y-axis tick labels
                range=[0, 100],
            ),
            plot_bgcolor="rgba(0,0,0,0)",  # Main background color
            paper_bgcolor="rgba(0,0,0,0)",  # Main background color
            font=dict(color="black"),  # Default black font for other elements
        )
        return fig
    except Exception as e:
        print(f"Error generating bar plot: {e}")
        return None
def generate_circular_progress(overall_score, background_color="#f8f3e8"):
    """
    Generates a circular progress chart with enhanced font styles and visuals.

    Args:
        overall_score (int): The overall score as a percentage (0-100).
        background_color (str): The background color for the plot.

    Returns:
        Plotly Figure: A circular progress chart.
    """
    fig = go.Figure()

    # Add circular progress pie
    fig.add_trace(go.Pie(
        values=[overall_score, 100 - overall_score],
        hole=0.7,
        direction='clockwise',
        sort=False,
        marker=dict(
            colors=['#4CAF50', '#E0E0E0']  # Green for progress, gray for remaining
        ),
        textinfo='none',  # No text on the pie chart
        hoverinfo='none'  # Disable hover information
    ))

    # Update layout for styling and customization
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{overall_score}%</b>",  # Bold percentage in the center
            x=0.5, y=0.5,
            font=dict(
                size=30,  # Larger font size for better visibility
                color="black",
                family="Verdana, Geneva, sans-serif"  # Clean and modern font style
            ),
            showarrow=False  # No arrow in the center
        )],
        plot_bgcolor="rgba(0,0,0,0)",  # Match app background
        paper_bgcolor="rgba(0,0,0,0)",  # Match app background
        showlegend=False,  # No legend
        margin=dict(t=0, b=0, l=0, r=0),  # Minimal margins for compact display
        height=300,  # Chart height
        width=300  # Chart width
    )

    return fig


def main():
    llm = initialize_model()
    set_background()

    # Generate individual feedback and cache it
    if "individual_feedback" not in st.session_state:
        st.session_state["individual_feedback"] = generate_individual(STATIC_QUESTIONS, llm)
    st.write(f"**Individual Feedback:** {st.session_state['individual_feedback']}")

    # Generate overall scores and cache them
    if "overall_scores" not in st.session_state:
        overall_scores_output = generate_individual_feedback(STATIC_QUESTIONS, llm)
        st.session_state["overall_scores"] = parse_scores(overall_scores_output)

    overall_scores = st.session_state["overall_scores"]

    if not overall_scores:
        st.error("Error: No overall scores generated")
    else:
        # Generate and display visualizations
        radar_fig = generate_radar_plot(overall_scores)
        bar_fig = generate_bar_plot(overall_scores)
        circular_fig = generate_circular_progress(overall_scores["Overall Score"])

        st.subheader("Performance Visualizations")
        col1, col2,col3 = st.columns(3)
        with col1:
            st.plotly_chart(radar_fig, use_container_width=True)
        with col2:
            st.plotly_chart(bar_fig, use_container_width=True)
        with col3:
        
            st.plotly_chart(circular_fig, use_container_width=True)


        

if __name__ == "__main__":
    main()
