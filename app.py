import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai.llms import GoogleGenerativeAI
import plotly.graph_objects as go
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
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
            color: brown;
        }
        .stApp {
            background: linear-gradient(135deg, #f8f3e8, #e9e0d1);
            color: black;
        }
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: black !important;
        }
        textarea, input[type="text"], button {
            background-color: white !important;
            color: black !important;
            border: 1px solid black !important;
            border-radius: 5px !important;
            padding: 10px;
        }
        textarea {
            height: 200px !important;
            width: 100% !important;
        }
        button {
            cursor: pointer;
            font-weight: bold;
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
# Initialize the Google GenAI LLM via LangChain with API key
def initialize_model():
    api_key =""
 # Replace with your actual API key
    return GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "insights" not in st.session_state:
    st.session_state.insights = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

# Generate dynamic questions based on input text
def generate_questions(input_text, llm):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in creating educational quiz questions that assess a spectrum of cognitive skills. Your task is to generate 10 balanced questions based on the provided text, distributed across Bloom's Taxonomy levels. 
context:{text}
Follow this distribution for cognitive skill levels:
- **Remembering:** 2 questions. Focus on recalling facts and basic concepts.
- **Understanding:** 2 questions. Focus on explaining ideas or concepts.
- **Applying:** 2 questions. Focus on using information in new situations.
- **Analyzing:** 2 questions. Focus on drawing connections among ideas.
- **Evaluating:** 1 question. Focus on justifying a decision or course of action.
- **Creating:** 1 question. Focus on producing new or original work.

Guidelines:
1. Questions should be clear, concise, and aligned with the specified cognitive skill levels.
2. Gradually increase complexity as you progress through the questions.
3. Ensure the questions test knowledge directly related to the provided text.
4.don't text like based on text kinda terms ,only need questions

Format the output as follows without " and ':
 Your question here for Remembering
 Your question here for Remembering
 Your question here for Understanding
 Your question here for Understanding
 Your question here for Applying
 Your question here for Applying
 Your question here for Analyzing
 Your question here for Analyzing
 Your question here for Evaluating
 Your question here for Creating
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    questions = chain.invoke({"text": input_text})
    return questions['text'].strip().split("\n")

# Generate overall feedback based on responses
def generate_overall_feedback( questions, responses, llm):
    prompt_template = PromptTemplate(
        input_variables=["text", "questions", "responses"],
        template=""" 
        Given the following:
        questions: {questions}
        And the student's responses to all the questions: {responses}
        
          You are a  teacher  tasked with evaluating the student's overall performance during an interactive session. Your goal is to assess the quality of their responses across multiple dimensions and provide a detailed evaluation against the question be more specific on student Consider the following,always be motivation and optimistic in your feedback, :

1. **Conceptual Understanding**: Did the student demonstrate accurate and thorough understanding of the topic? Were there significant misconceptions or gaps in knowledge?
2. **Clarity and Coherence**: Were the responses logically structured, well-organized, and easy to follow? Identify any areas of confusion.
3. **Depth of Knowledge**: Did the student provide in-depth insights, or did their responses remain at a surface level? Highlight if advanced aspects were inadequately addressed.
4. **Application and Examples**: Did the student include relevant real-world examples or practical applications? Were these examples appropriate and accurate?
5. **Accuracy and Precision**: Were the responses factually correct and free from ambiguities or errors? Flag any notable inaccuracies.
6. **Critical Thinking**: Did the responses demonstrate reasoning, analysis, or evaluation? Did the student provide evidence or justification for their arguments?
7. **Engagement with Feedback**: How effectively did the student incorporate feedback or follow-up prompts into their responses? Were adjustments meaningful and noticeable?
8. **Creativity and Originality**: Did the responses showcase innovative ideas, unique perspectives, or creative approaches to the topic?
9. **Relevance**: Did the student stay focused on the topic and directly address the questions posed, without unnecessary deviations?
10. **Communication Skills**: Was the language, grammar, and syntax appropriate for the context? Highlight any issues affecting clarity or professionalism.

After analyzing the student's overall responses, provide a detailed evaluation in the following format and in bulletin points,only need these below three sections no addtional texts:


    1. **Your Superpowers 💪🔥**:
        - Highlight specific strengths such as understanding ,application of knowledge, and engagement with the topic.
        - This section should be at least 400 characters and at most 500 characters,should be  in bullet points not as paragraph

    2. **Where We Can Level Up 🎯🔨**:
        - Identify gaps in knowledge and lack of specificity. Mention areas like conceptual understanding, real-world applications, and use of precise terminology.
        - This section should be at least 400 characters and at most 500 characters,should be  in bullet points not as paragraph

    3. **Your Journey to Mastery 🏅🚀**:
        - Focus on helping the student develop depth of knowledge by studying more advanced topics or working on real-world examples.
        - Suggest improving the clarity of responses and the specificity of examples.
        - This section should be at least 400 characters and at most 500 characters,should be  in bullet points not as paragraph

        
        Be constructive and supportive in your feedback. Write your response in a clear and concise manner.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({"questions": "\n".join(questions), "responses": "\n".join(responses)})
    return result['text'].strip()

# Generate an overall grade based on responses
def generate_overall_grade( questions, responses, llm):
    prompt_template = PromptTemplate(
        input_variables=["questions", "responses"],
        template="""
         Evaluate the student's response to the question provided,Your task is to evaluate a student's response to a question,the output should be only 10 scores and no texts:
        Given the following : 
        Question: {questions}
        Response: {responses}
    
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
    result = chain.invoke({ "questions": "\n".join(questions), "responses": "\n".join(responses)})
    return result['text'].strip()
# Refine the overall feedback using another LLM

    # Create a chain with the LLM model and refined prompt

def generate_individual_feedback(questions, responses, llm):
    prompt_template = PromptTemplate(
        input_variables=["questions", "responses"],
        template="""
         
        Question: {questions}
        Response: {responses}
        
       You are a highly skilled assistant focused on evaluating user responses in text-based Q&A tasks. Your job is to generate a **performance table** for 10 questions, displaying scores and feedback for the following metrics: 

1. 🎯 **Relevance **: How well the response aligns with the question's context and requirements.
2. ✅ **Accuracy**: The correctness and factuality of the response.
3. 💡 **Clarity **: The simplicity, coherence, and ease of understanding in the answer.
4. 🌊 **Depth **: The level of detail, explanation, and elaboration in the response.

### **Your task:**
1. For each question, provide a **detailed performance table row** with:
   - The question text.
   - Scores (in percentages) for Relevance, Accuracy, Clarity, and Depth.
   - Fun ,motivating,postive and intuitive face expersion like emoji-based tags like to make feedback engaging.
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
    feed = chain.invoke({"questions": questions, "responses": responses})
    return feed['text'].strip()
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
                text="Radar Chart of Performance Metrics",
                font=dict(size=20, color="black"),  # Title styling
                x=0.2,  # Center align
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
                text="Performance Metrics ",
                font=dict(size=20, color="black"),  # Title styling with black font
                x=0.2,  # Center alignment
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
        annotations=[
           
            dict(
                text="Overall Score",  # New text annotation
                x=0.2,  # Position below the chart
                font=dict(
                    size=16,
                    color="gray",
                    family="Verdana, Geneva, sans-serif"
                ),
                showarrow=False
            ),
        ],
        plot_bgcolor="rgba(0,0,0,0)",  # Match app background
        paper_bgcolor="rgba(0,0,0,0)",  # Match specified background
        showlegend=False,  # No legend
        margin=dict(t=0, b=0, l=0, r=0),  # Minimal margins for compact display
         # Chart width
    )
    return fig


   
# Streamlit app 
 



# Initialize LLM
set_background()
llm = initialize_model()
if "page" not in st.session_state:
    st.session_state.page = "input"
if "questions" not in st.session_state:
    st.session_state.questions = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
def set_page_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f8f3e8, #e9e0d1);
            color: black;
        }
        .chat-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question-box {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .response-box {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .metrics-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def init_model():
    api_key = "YOUR_API_KEY"  # Replace with actual API key
    return GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
import PyPDF2
def show_input_page():
    st.title("📝 Personalized Quiz Generation with Evaluation and Feedback Using AI")
    st.markdown("""
        <div class="chat-container">
            <h3>Hey there, future quiz champion! 🎉</h3>
            <p>I’m your AI buddy, here to make quizzes smarter, feedback sharper, and learning way more fun! Let’s dive into this adventure together.</p>
            <ul>
                <li>💡 Scribble down your own text.</li>
                <li>📄 Upload a PDF treasure trove.</li>
                <li>📹 Or drop in a YouTube link for some video magic.</li>
            </ul>
            <p>Let’s turn your knowledge into quizzes and your quizzes into success! 🚀</p>
        </div>
    """, unsafe_allow_html=True)

    # Text input area
    input_text = st.text_area("Enter your text here:", height=200)

    # File uploader for PDF
    uploaded_file = st.file_uploader("Or upload a PDF document:", type=["pdf"])

    # YouTube link input
    youtube_url = st.text_input("Or provide a YouTube video link for transcription:")

    if uploaded_file is not None:
        try:
            # Extract text from the PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()
            st.session_state.input_text = extracted_text
            st.success("Text extracted from the uploaded PDF successfully!")
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
            return

    if youtube_url:
        try:
            # Directly use the YouTube URL to fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(youtube_url)

            # Format transcript to text
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript)

            # Store the transcript text in session state
            st.session_state.input_text = transcript_text
            st.success("Transcript fetched successfully from YouTube video!")
        except ValueError as ve:
            st.error(f"Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred while fetching the transcript: {e}")
            return

    # Combine input_text and extracted_text if both exist
    final_text = input_text.strip()
    if "input_text" in st.session_state:
        final_text = (final_text + " " + st.session_state.input_text).strip()

    # Start Quiz button
    if st.button("Start Quiz", type="primary"):
        if final_text:
            st.session_state.input_text = final_text

            # Generate questions using the LLM
            all_questions = generate_questions(final_text, llm)

            # Filter valid questions
            valid_questions = [q for q in all_questions if q.strip()]
            st.session_state.questions = valid_questions  # Only store valid questions

            if not valid_questions:
                st.error("No valid questions could be generated. Please try different input text.")
            else:
                st.session_state.page = "quiz"
                st.experimental_rerun()
        else:
            st.error("Please enter some text, upload a PDF, or provide a YouTube link before starting the quiz.")



# Modified show_quiz_page function
def show_quiz_page():
    st.title(" Quiz Session📝")

    if not st.session_state.questions:
        st.error("No valid questions available. Please go back and provide valid input.")
        return

    # Display the chat-like history of previous questions and answers
    chat_placeholder = st.container()  # Placeholder for the chat interface

    # Display the current question
    progress = st.session_state.current_index / len(st.session_state.questions)
    st.progress(progress)

    with st.container():
        st.markdown(f"""
            <div class="question-box">
                <h4>Question {st.session_state.current_index + 1} of {len(st.session_state.questions)}</h4>
                <p>{st.session_state.questions[st.session_state.current_index]}</p>
            </div>
        """, unsafe_allow_html=True)

        # Save the user's response
        user_response = st.text_area("Your Answer:", height=150, key=f"response_{st.session_state.current_index}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Submit Answer", type="primary"):
                if user_response.strip():
                    # Append the response and update the chat history
                    st.session_state.responses.append(user_response)
                    st.session_state.chat_history.append((st.session_state.questions[st.session_state.current_index], user_response))

                    if st.session_state.current_index + 1 < len(st.session_state.questions):
                        st.session_state.current_index += 1
                    else:
                        st.session_state.page = "results"
                    st.experimental_rerun()
                else:
                    st.error("Please provide an answer before submitting.")

    # Render the chat history dynamically
    with chat_placeholder:
        if st.session_state.chat_history:
            st.markdown("<h4>Chat History</h4>", unsafe_allow_html=True)
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>Q{i + 1}:</strong> {q}</p>
                        <p><strong>A:</strong> {a}</p>
                    </div>
                """, unsafe_allow_html=True)


# Modified show_results_page function
def show_results_page():
    st.title(" Quiz Feedback & Analysis📊")

    # Use st.session_state.responses and st.session_state.questions
    if not st.session_state.responses:
        st.error("No responses recorded. Please complete the quiz first.")
        return

    # Example: Display the responses
   

    # Generate analysis using responses and questions
    if "overall_scores" not in st.session_state:
        overall_scores_output = generate_overall_grade(st.session_state.questions, st.session_state.responses,llm)
        st.session_state["overall_scores"] = parse_scores(overall_scores_output)

    overall_scores = st.session_state["overall_scores"]

    if not overall_scores:
        st.error("Error: No overall scores generated")
        return

    radar_fig = generate_radar_plot(overall_scores)
    bar_fig = generate_bar_plot(overall_scores)
    circular_fig = generate_circular_progress(overall_scores["Overall Score"])

    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(circular_fig, use_container_width=True)
    with col2:
        st.plotly_chart(bar_fig, use_container_width=True)
    with col3:
        st.plotly_chart(radar_fig, use_container_width=True)

    if "individual_feedback" not in st.session_state:
        st.session_state["individual_feedback"] = generate_individual_feedback(st.session_state.questions,st.session_state.responses, llm)
    st.write(f"**Individual Feedback:** {st.session_state['individual_feedback']}")

    if "overall_feedback" not in st.session_state:
        st.session_state["overall_feedback"] = generate_overall_feedback(st.session_state.questions,st.session_state.responses, llm)
    st.write(f"**Overall Feedback:** {st.session_state['overall_feedback']}")

    if st.button("Start New Quiz"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.page = "input"
        st.experimental_rerun()
def main():
    if "page" not in st.session_state:
        st.session_state.page = "input"
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.page == "input":
        show_input_page()
    elif st.session_state.page == "quiz":
        show_quiz_page()
    elif st.session_state.page == "results":
        show_results_page()

if __name__ == "__main__":
    main()
