import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Utility functions
def get_text_chunks(text):
    """Splits the text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up the question-answering conversational chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, respond with 'Answer not available in the context'.\n
    Context: {context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_input(user_question):
    """Processes the user input for the chatbot and returns the response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_pdf_text(file_path):
    """Extracts text from the provided PDF file."""
    reader = PdfReader(file_path)
    raw_text = "".join([page.extract_text() for page in reader.pages])
    return raw_text

# Section components
def render_about_section():
    """Renders the 'About Me' section."""
    st.balloons()
    st.header("👨‍💻 About Me")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.header(" Who am I? ")
        st.write("""
        I'm a highly skilled software engineer with experience in AI, machine learning, big data, cloud computing, and full-stack development.
        """)
    with col3: 
        st.image("computer-science.png")

def render_skills_section():
    """Renders the 'Skills' section."""
    st.header("🔧 Skills")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1,1,1])
    with col1:
        st.header(":shamrock: Programming")
        st.write("Python, Golang, Java, SQL, JavaScript, C")

    with col2:
        st.header(":shamrock: Backend Development")
        st.write("NodeJS, AngularJS, ExpressJS, ReactJS, Vue.js, Django")

    with col3:
        st.header(":shamrock: Backend Development")
        st.write("NodeJS, AngularJS, ExpressJS, ReactJS, Vue.js, Django")

    with col4:
        st.header(":shamrock: Cloud Development")
        st.write("AWS, GCP, Azure")

    with col5:
        st.header(":shamrock: Data")
        st.write("Docker, Kubernetes, Spark, Hadoop")


def render_experience_section():
    """Renders the 'Work Experience' section."""
    st.header("💼 Work Experience")
    experience_list = [
        {
            "role": "Software Engineer Intern - Nutanix",
            "duration": "May 2024 – Present",
            "description": [
                "Developed Distributed Tracing using open-source Jaeger and OpenTelemetry.",
                "Analyzed service dependencies in microservices architecture.",
                "Enhanced querying with Elasticsearch."
            ]
        },
        {
            "role": "Senior Data Engineer - LTIMindtree",
            "duration": "Jul 2021 – Jul 2023",
            "description": [
                "Led Azure Synapse data warehouse development.",
                "Secured $36 million in revenue.",
                "Decreased BI report load times by 85%."
            ]
        }
    ]
    for exp in experience_list:
        st.subheader(exp["role"])
        st.write(f"**Duration:** {exp['duration']}")
        st.write("\n".join(f"- {item}" for item in exp['description']))

def render_projects_section():
    """Renders the 'Projects' section."""
    st.header("📊 Projects")
    projects = [
        {
            "title": "Personality Prediction System",
            "description": "Led a team to build a system for predicting employee personalities.",
            "publication": "[Publication](https://link.springer.com/chapter/10.1007/978-981-99-5354-7_15)"
        },
        {
            "title": "Ride Insights",
            "description": "Engineered data insights for NYC taxi trip records.",
        }
    ]
    for project in projects:
        st.subheader(project["title"])
        st.write(f"- {project['description']}")
        if 'publication' in project:
            st.markdown(f"🔗 {project['publication']}")

def render_education_section():
    """Renders the 'Education' section."""
    st.header("🎓 Education")
    education_list = [
        {"degree": "Master of Science in Computer Science", "institution": "San Jose State University", "duration": "Aug 2023 – May 2025", "gpa": "GPA – 3.83/4.0"},
        {"degree": "Bachelor of Science in Computer Engineering", "institution": "University of Mumbai", "duration": "Aug 2017 - May 2021"}
    ]
    for edu in education_list:
        st.subheader(edu["degree"])
        st.write(f"**Institution:** {edu['institution']}")
        st.write(f"**Duration:** {edu['duration']}")
        if 'gpa' in edu:
            st.write(f"**GPA:** {edu['gpa']}")


def add_custom_css():
    """Adds custom CSS for styling the banner and footer."""
    st.markdown(
        """
        <style>
         [data-testid=stSidebar] {
        background-color: #0a0909;
        color:white
    }
        .banner {
            background-color: #4CAF50;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        # .footer {
        #     background-color: #FF5733;
        #     padding: 10px;
        #     color: white;
        #     text-align: center;
        #     font-size: 18px;
        #     position: fixed;
        #     width: 100%;
        #     bottom: 0;
        # }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to display photo in the sidebar
def render_photo():
    """Displays the user's photo in the sidebar."""
    image = Image.open("Kalindi_Vijesh_Parekh_Profile.jpg")  # Replace with the actual path to your image file
    st.sidebar.image(image, caption="Mihir Dhirajlal Satra", use_column_width=True)
   
def render_header():
    """Displays a header saying 'Hi, I am Mihir' at the top of every page."""
    st.markdown('<h1 style="text-align:center;">Hi, I am Mihir!</h1>', unsafe_allow_html=True)

# Main application
def main():
    st.set_page_config(page_title="Mihir Dhirajlal Satra's Resume", page_icon=":briefcase:", layout="wide")

    add_custom_css()

    # Banner
    #st.markdown('<div class="banner">I am Mihir</div>', unsafe_allow_html=True)

    # Render the header
    render_header()

    st.sidebar.title("Navigation")
    st.sidebar.markdown('<h1 style="text-align:center; color:white">Mihir Dhirajlal Satra - MS CS Graduate</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("📅 Graduating in May 2025")
    
    # Embed photo 
    # Display the user's photo
    render_photo()

    

    # Navigation
    section = st.sidebar.selectbox("Select a section to view:", ("About", "Skills", "Work Experience", "Projects", "Education"))

    # Display section
    if section == "About":
        render_about_section()
    elif section == "Skills":
        render_skills_section()
    elif section == "Work Experience":
        render_experience_section()
    elif section == "Projects":
        render_projects_section()
    elif section == "Education":
        render_education_section()

    # Chatbot section
    st.header("💬 Chatbot")
    user_question = st.text_input("You: ", placeholder="Type your message here...")
    if user_question:
        file_path = r"C:\Users\Checkout\Documents\Kalindi's_Documents\Semester_3\Cs 256\ResumePortfolio\MihirDhirajlal_Satra_Resume.pdf"  
        raw_text = get_pdf_text(file_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        response = handle_user_input(user_question)
        st.write("Chatbot Reply: ", response)

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.write("© 2024 Mihir Dhirajlal Satra's Resume Website | Powered by Streamlit")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

