import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai.llms import GoogleGenerativeAI

# Initialize the Google GenAI LLM via LangChain with API key
def initialize_model():
    api_key = "your api key"  # Replace with your actual API key
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
STATIC_QUESTIONS = [
    "What are the primary components of the Earth's atmosphere?",
    "Explain the process of photosynthesis in plants.",
    "How do viruses differ from bacteria?",
    "What is the significance of Newton's three laws of motion?",
    "Describe the water cycle and its stages.",
    "Explain the difference between renewable and non-renewable energy sources.",
    "What are the key characteristics of a healthy ecosystem?",
    "Analyze the factors that contribute to climate change.",
    "Evaluate the impact of deforestation on biodiversity.",
    "Create a plan for reducing plastic waste in your community."
]

# Generate overall feedback based on responses
def generate_overall_feedback(input_text, questions, responses, llm):
    prompt_template = PromptTemplate(
        input_variables=["text", "questions", "responses"],
        template=""" 
        Given the following text: {text}
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

After analyzing the student's overall responses, provide a detailed evaluation in the following format and in bulletin points:

### **Strengths**

- Example:
  - **Conceptual Understanding**: The student demonstrated excellent understanding of [specific topic], with a clear and thorough explanation of key principles.
  - **Clarity and Coherence**: Responses were well-structured, logically organized, and easy to follow.

### **Weaknesses**
- Clearly identify **key weaknesses**, using bold text to emphasize critical issues (e.g., **lack of depth**, **inaccurate examples**, **insufficient analysis**).
- Example:
  - **Depth of Knowledge**: The student’s responses often remained surface-level and did not address more advanced aspects of [specific topic].
  - **Critical Thinking**: Responses lacked sufficient analysis or justification, with limited evidence provided to support arguments.

### **Areas for Improvement**
- Provide actionable steps to address weaknesses, highlighting **priority areas** (e.g., **critical thinking**, **accuracy**, **real-world application**) in bold for emphasis.
- Example:
  - **Application and Examples**: Focus on incorporating accurate and relevant real-world examples to demonstrate understanding.
  - **Critical Thinking**: Practice providing evidence-based reasoning and justify conclusions with supporting details.

Write your evaluation in a structured, concise, and encouraging tone. Highlight all critical points in **bold** to ensure clarity and focus.
        
        Be constructive and supportive in your feedback. Write your response in a clear and concise manner.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({"text": input_text, "questions": "\n".join(questions), "responses": "\n".join(responses)})
    return result['text'].strip()

# Generate an overall grade based on responses
def generate_overall_grade(input_text, questions, responses, llm):
    prompt_template = PromptTemplate(
        input_variables=["text", "questions", "responses"],
        template="""
         Evaluate the student's response to the question provided:
        Given the following text: {text}context
        Question: {questions}
        Response: {responses}

        Use the following criteria to evaluate be  specific on student and avoid generalization like general practice 

        - **Knowledge and Understanding**: Does the response demonstrate a comprehensive understanding of the topic? Does it explain key concepts with depth and accuracy?
        - **Depth and Detail**: Does the response include relevant facts, examples, and explanations that develop the ideas fully? Is it clear and detailed, or does it feel superficial?
        - **Clarity and Organization**: Is the response well-organized, logically structured, and easy to follow?
        - **Accuracy**: Is the information presented factually correct, and are the relationships between concepts explained properly?
        Based on these criteria, assign a grade using this rubric:
        - **A (90-100)**: Excellent. Shows deep understanding, strong development of ideas, clear organization, and accurate information.
        - **B (80-89)**: Good. Demonstrates solid understanding, but may lack full depth or clarity in some areas.
        - **C (70-79)**: Satisfactory. Shows basic knowledge but lacks depth or is somewhat unclear.
        - **D (60-69)**: Below average. Limited understanding with significant gaps or inaccuracies.
        - **F (0-59)**: Inadequate. Irrelevant, incorrect, or insufficient content.

        Evaluate the given answer and assign a grade (A, B, C, D, F) based on its quality. Only return the grade followed by 2–4 words inspired by popular Tamil movie dialogues.

For high scores (A and B): Use iconic, celebratory Tamil movie lines that reflect success and victory.
For medium scores (C): Use neutral yet motivational Tamil movie dialogues to highlight progress and potential.
For low scores (D and F): Use only  Tamil movie dialogues  and a comeback (strictly avoid sarcasm or discouragement).use postive and fun emojis 
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({"text": input_text, "questions": "\n".join(questions), "responses": "\n".join(responses)})
    return result['text'].strip()
# Refine the overall feedback using another LLM
def refine_overall_feedback(feedback, llm, questions, responses):
    refine_prompt_template = PromptTemplate(
        input_variables=["feedback", "questions", "responses"],
        template=""" 
        You are a fun and motivating teacher who use uses lots of emojis to engage with students who tasked with refining the feedback for a student's performance. The feedback should be encouraging, concise, and clear. Ensure the feedback is well-organized, flows logically, and highlights key points effectively. Focus more on the student's performance and knowledge, and avoid generalizations like "general practice.be specific on student ,always be motivation and optimistic in your feedback, :


        Feedback to refine:
        {feedback}

        Question: {questions}
        Response: {responses}
        provide a detailed evaluation in the following format and in bulletin points:

        1. **Strengths**:
            - Highlight specific strengths such as **understanding**, **application of knowledge**, and **engagement with the topic**. 
            - This section should be at least 400 characters and at most 500 characters.

        2. **Weaknesses**:
            - Identify **gaps in knowledge** and **lack of specificity**. Mention areas like **conceptual understanding**, **real-world applications**, and **use of precise terminology**.
            - This section should be at least 400 characters and at most 500 characters.

        3. **Areas of Improvement**:
            - Focus on helping the student develop **depth of knowledge** by studying more advanced topics or working on real-world examples.
            - Suggest improving the **clarity** of responses and the **specificity** of examples.
            - This section should be at least 400 characters and at most 500 characters.
         ** End with a motivational movie dialouge based on student performance comment and emojis to encourage the student. only give dialouge with emoji don't add add anything else**
        Refine the following feedback to ensure it is polished, concise, motivating, and actionable. Focus on providing actionable insights while maintaining a constructive and positive tone.
        """
    )
    # Create a chain with the LLM model and refined prompt
    chain = LLMChain(llm=llm, prompt=refine_prompt_template)
    
    # Pass the feedback, questions, and responses into the chain to generate the refined feedback
    refined_feedback = chain.invoke({"feedback": feedback, "questions": questions, "responses": responses})
    
    # Return the refined feedback text
    return refined_feedback['text'].strip()
def generate_individual_feedback(question, response, llm):
    prompt_template = PromptTemplate(
        input_variables=["question", "response"],
        template="""
         You are a fun and motivating teacher who use uses lots of emojis to  tasked with evaluating a student's response to a specific question, be optimistic and motivational in your feedback
        Question: {question}
        Response: {response}
        
        Provide detailed feedback on:
        1. Relevance: Did the response directly address the question?
        2. Accuracy: Was the information factually correct?
        3. Clarity: Was the response well-structured and easy to understand?
        4. Depth: Did the response show insight or go beyond surface-level information?
        
        Format your feedback in ths format each in 1-2 lines and in bullet points :
        - **Relevance:** Your feedback here.
        - **Accuracy:** Your feedback here.
        - **Clarity:** Your feedback here.
        - **Depth:** Your feedback here.bn/ 

        **Ideal Answer**: ideal answer for question(it can be a paragraph or a few lines of max 500 characters)
       
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    feed = chain.invoke({"question": question, "response": response})
    return feed['text'].strip()

# Streamlit app 
st.title("Interactive Quiz and Feedback App")

# Step 1: Upload or Input Text
st.header("Step 1: Upload or Input Text")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
input_text = ""
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
else:
    input_text = st.text_area("Or paste your text below:", height=200)

if input_text:
    st.success("Text successfully uploaded or input.")

# Initialize session state
if "questions" not in st.session_state:
   st.session_state.questions = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LLM
llm = initialize_model()

# Generate questions if not already generated
if not st.session_state.questions and input_text:
    st.session_state.questions =STATIC_QUESTIONS
    print(st.session_state.questions)

# Step 2: Interactive Q&A Session
if st.session_state.current_index < len(st.session_state.questions):
    # Filter out empty or invalid questions
    valid_questions = [q for q in st.session_state.questions if q.strip()]
    st.session_state.questions = valid_questions  # Reassign only valid questions

    # Ensure the current_index does not exceed the number of valid questions
    if st.session_state.current_index >= len(valid_questions):
        st.session_state.current_index = len(valid_questions) - 1

    # Display chat history if available
    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, (question, response) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i + 1}:** {question}")
            st.write(f"**A{i + 1}:** {response}")

    # Get the current question
    current_question = st.session_state.questions[st.session_state.current_index]

    # Combine insight and question in a single text
    if st.session_state.current_index > 0:
        last_question = st.session_state.questions[st.session_state.current_index - 1]
        last_response = st.session_state.responses[-1]
        st.write(
            f"#### Let's move on to the next question:\n\n"
            f"**{st.session_state.current_index + 1}: {current_question}**"
        )
    else:
        st.write(f"### {st.session_state.current_index + 1}: {current_question}**")

    # Capture user response
    user_response = st.text_input("Your Answer:", key=f"response_{st.session_state.current_index}")

    if st.button("Submit Answer"):
        if user_response:
            st.session_state.responses.append(user_response)
            st.session_state.chat_history.append((current_question, user_response))
            st.session_state.current_index += 1
            st.experimental_rerun()
        else:
            st.error("Please provide an answer before submitting.")
else:
    # Ensure no invalid or empty questions are processed
    valid_questions = [q for q in st.session_state.questions if q.strip()]
    st.session_state.questions = valid_questions  # Reassign only valid questions

    # Step 3: Overall Feedback and Grading
st.header("Feedback and Grade")

# Loop through each question-response pair


# Step 4: Overall Feedback and Grade (Only after all individual feedback is displayed)


if len(st.session_state.responses) == len(st.session_state.questions):
    # Generate individual feedback for each question-response pair
    
    for i, (question, response) in enumerate(zip(st.session_state.questions, st.session_state.responses)):
        if f"individual_feedback_{i}" not in st.session_state:
        # Generate individual feedback for each question-response pair
            feedback = generate_individual_feedback(question, response, llm)
            st.session_state[f"individual_feedback_{i}"] = feedback  # Save feedback to session state

    # Display question, response, and individual feedback
        st.write(f"### Question {i + 1}: {question}")
        st.write(f"**Your Answer:** {response}")
        st.write(f"**Feedback:** {st.session_state[f'individual_feedback_{i}']}")
    if "overall_feedback" not in st.session_state:
        st.write("Generating overall feedback...")
        overall_feedback = generate_overall_feedback(input_text, st.session_state.questions, st.session_state.responses, llm)

        # Refine the overall feedback and store it in session state
        refined_feedback = refine_overall_feedback(overall_feedback, llm, st.session_state.questions, st.session_state.responses)
        st.session_state.overall_feedback = refined_feedback

    # Display refined overall feedback
    st.subheader("Overall Feedback")
    st.write(st.session_state.overall_feedback)

    
    st.write("Generating overall grade...")
    overall_grade = generate_overall_grade(input_text, st.session_state.questions, st.session_state.responses, llm)
    st.session_state.overall_grade = overall_grade
    st.subheader("Overall Grade")
    st.write(st.session_state.overall_grade)

    # Display overall grade
