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
        Hi! I'm Mihir, a tech enthusiast currently mastering the art of distributed computing and AI at San Jose State University. When I’m not coding in Python or wrestling with Kubernetes, you can find me optimizing data processes (with a sprinkle of Golang and Elasticsearch) to make systems run faster and smarter. I've helped boost system performance, slashed storage costs, and made data visualization my playground!
        Whether I’m building dashboards, automating workflows, or crafting cloud solutions, I’m always on the lookout for new ways to make tech do more, with less effort. Oh, and I love a good challenge—especially if it involves solving complex problems or breaking down a tricky algorithm. Let's just say I thrive on optimization—whether it's data pipelines or my morning coffee routine! 😄
        """)
    with col3: 
        st.image("computer-science.png")
        

def render_skills_section():
    """Renders the 'Skills' section."""
    st.header("🔧 Skills")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1,1,1])
    with col1:
        st.header(":shamrock: Programming")
        lst =['Python', 'Golang', 'Java', 'SQL', 'JavaScript', 'C']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)
       

    with col2:
        st.header(":shamrock: Frontend Development")
        lst = ['NodeJS', 'AngularJS', 'ExpressJS', 'ReactJS', 'Vue.js', 'Django']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)
    

    with col3:
        st.header(":shamrock: Backend Development")
        lst = ['Flask' 'Django', 'SpringBoot']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)

    with col4:
        st.header(":shamrock: Cloud Development")
        lst = ['AWS' 'GCP', 'Azure']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)

    with col5:
        st.header(":shamrock: Data")
        lst = ['Docker', 'Kuberentes', 'Spark']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)

def render_experience_section():
    """Renders the 'Work Experience' section."""
    st.header("💼 Work Experience")

    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust the number of columns depending on how much content you have

    with col1:
        st.header(":male-office-worker: Software Engineer Intern - Nutanix")
        st.markdown("**Duration:** May 2024 – Present")
        nutanix_desc = [
            "Developed Distributed Tracing using open-source Jaeger and OpenTelemetry.",
            "Analyzed service dependencies in microservices architecture.",
            "Enhanced querying with Elasticsearch."
        ]
        desc_str = ''
        for desc in nutanix_desc:
            desc_str += "- " + desc + "\n"
        st.markdown(desc_str)

    with col2:
        st.header(":male-office-worker: Senior Data Engineer - LTIMindtree")
        st.markdown("**Duration:** Jul 2021 – Jul 2023")
        ltimindtree_desc = [
            "Led Azure Synapse data warehouse development.",
            "Secured $36 million in revenue.",
            "Decreased BI report load times by 85%."
        ]
        desc_str = ''
        for desc in ltimindtree_desc:
            desc_str += "- " + desc + "\n"
        st.markdown(desc_str)

    with col3:
        st.header(":male-office-worker: Software Engineer Intern - GRT Global Logistics")
        st.markdown("**Duration:** Dec 2019 – Jan 2020")
        grt_desc = [
            "Streamlined ERP software testing with Selenium automation.",
            "Collaborated with the technical team to enhance system design."
        ]
        desc_str = ''
        for desc in grt_desc:
            desc_str += "- " + desc + "\n"
        st.markdown(desc_str)


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

    # Define the columns
    col1, col2 = st.columns([1, 1])  # We can use 2 columns, but you can adjust it based on your preferences

    with col1:
        st.header(":mortar_board: Master's Degree")
        st.markdown("**Degree:** Master of Science in Computer Science")
        st.markdown("**Institution:** San Jose State University")
        st.markdown("**Duration:** Aug 2023 – May 2025")
        st.markdown("**GPA:** 3.83/4.0")

    with col2:
        st.header(":mortar_board: Bachelor's Degree")
        st.markdown("**Degree:** Bachelor of Science in Computer Engineering")
        st.markdown("**Institution:** University of Mumbai")
        st.markdown("**Duration:** Aug 2017 - May 2021")

def render_gallery_section():
    """Displays a gallery of candid photos."""
    st.header("📸 Candid Photo Gallery")

    # Path to the folder where the photos are stored
    photos_folder = "gallery"

    # Get all image files in the folder
    image_files = [f for f in os.listdir(photos_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Display images in a grid
    cols = st.columns(3)  # Adjust the number of columns as needed
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(photos_folder, image_file)
        image = Image.open(image_path)
        with cols[idx % 3]:
            st.image(image, use_column_width=True)

# Function to render the contact section
def render_contact_section():
    """Renders the 'Contact Me' section with social media links and a contact form."""
    st.header("📬Contact Me")

    # Social Media Links
    st.write("Feel free to connect with me on my social media:")
    col1, col2, col3 = st.columns(3)
    st.markdown("""
        <style>
        .icon-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .icon-container a {
            margin: 0 20px;
            font-size: 50px;
            text-decoration: none;
        }
        </style>
        <div class="icon-container">
            <a href="https://www.linkedin.com/in/mihirsatra/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="50" height="50">
            </a>
            <a href="https://github.com/mihirsatra44" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="50" height="50">
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    # with col1:
    #     st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/mihirsatra/)")
    # with col2:
    #     st.markdown("[![Gmail](https://img.shields.io/badge/Gmail-red)](mailto:kparekh305@gmail.com)")
    # with col3:
    #     st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-black)](https://github.com/KP-305)")

    # Contact Form
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        query = st.text_area("Your Query")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.success(f"Thank you, {name}! Mihir will get back to you soon.")

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
    image = Image.open("Mihir.jpg")  # Replace with the actual path to your image file
    st.sidebar.image(image, caption="Mihir Dhirajlal Satra", use_column_width=True)
   
def render_header():
    """Displays a header saying 'Hi, I am Mihir' at the top of every page."""
    st.markdown('<h1 style="text-align:center;">Hi, I am Mihir!</h1>', unsafe_allow_html=True)

# Function to render the download PDF section
def render_download_pdf_section():
    """Renders the 'Download PDF' section with a button to download Mihir's resume."""
    st.header("📄 Download Mihir's Resume")

    # Path to the PDF file
    pdf_file_path = "resumepdf/Mihir_Dhirajlal_Satra_Resume.pdf"  # Replace with the actual path

    with open(pdf_file_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # Download button for the PDF file
    st.download_button(label="Download Mihir's Resume", 
                       data=pdf_bytes, 
                       file_name="Mihir_Dhirajlal_Satra_Resume.pdf", 
                       mime="application/pdf")

def render_awards_and_honors():
   
    st.title("🏆 Awards and Honors")
    # Create a container for better layout
    with st.container():
        col1, col2 = st.columns([1, 3])  # Two columns: one narrow for icons, another for text
        
        # Example award 1
        with col1:
            st.image("https://img.icons8.com/?size=100&id=GUqqHpQDjGqG&format=png&color=000000", width=50)  # Placeholder for award icon
        with col2:
            st.markdown("**Dean's List** (2021)")
            st.markdown("*Awarded for maintaining a GPA of 3.8 or higher at KJ Somaiya.*")

        st.markdown("---")  # Horizontal line to separate awards
        
        # Example award 2
        with col1:
            st.image("https://img.icons8.com/color/50/000000/cloud.png", width=50)  # Placeholder for award icon
        with col2:
            st.markdown("**AWS Certified Solutions Architect** (2022)")
            st.markdown("*Certification awarded for expertise in cloud architecture and AWS technologies.*")
        
        st.markdown("---")  # Horizontal line to separate awards

        # Example award 3
        with col1:
            st.image("https://img.icons8.com/color/50/000000/trophy.png", width=50)  # Placeholder for award icon
        with col2:
            st.markdown("**Top Performer Award** (2022)")
            st.markdown("*Recognized for outstanding performance at LTI.*")

        # Add more awards as needed
        st.markdown("---")  # Horizontal line to separate awards

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
    section = st.sidebar.selectbox("Select a section to view:", ("About", "Skills", "Work Experience", "Projects", "Education", "Gallery","Contact Me", "Download Resume","Awards and Honors"))

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
    elif section == "Gallery":
        render_gallery_section()
    elif section == "Contact Me":
        render_contact_section()
    elif section == "Download Resume":
        render_download_pdf_section()
    elif section == "Awards and Honors":
        render_awards_and_honors()

    with st.container(border = True):
    # Chatbot section
        st.header(":balloon: Ask Me Anything!!")
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

