import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = st.secrets["api_key"]
genai.configure(api_key=GOOGLE_API_KEY)

# Database setup
def init_db():
    conn = sqlite3.connect('resume.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS skills
                 (category TEXT, skill TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS experience
                 (title TEXT, company TEXT, duration TEXT, description TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (title TEXT, description TEXT, publication TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS education
                 (degree TEXT, institution TEXT, duration TEXT, gpa TEXT)''')
    conn.commit()
    return conn


def populate_db(conn):
    c = conn.cursor()
    
    # Skills
    skills = [
        ('Programming', 'Python'), ('Programming', 'Golang'), ('Programming', 'Java'),
        ('Frontend', 'ReactJS'), ('Frontend', 'AngularJS'), ('Frontend', 'Vue.js'),
        ('Backend', 'Django'), ('Backend', 'Flask'), ('Backend', 'SpringBoot'),
        ('Cloud', 'AWS'), ('Cloud', 'GCP'), ('Cloud', 'Azure'),
        ('Data Technologies', 'Docker'), ('Data Technologies', 'Kubernetes'), ('Data Technologies', 'Spark')
    ]
    c.executemany('INSERT OR REPLACE INTO skills VALUES (?,?)', skills)
    
    # Experience
    experiences = [
        ('Software Engineer Intern', 'Nutanix', 'May 2024 – Present', 
         'Developed Distributed Tracing using open-source Jaeger and OpenTelemetry.'),
        ('Senior Data Engineer', 'LTIMindtree', 'Jul 2021 – Jul 2023', 
         'Led Azure Synapse data warehouse development.'),
        ('Software Engineer Intern', 'GRT Global Logistics', 'Dec 2019 – Jan 2020', 
         'Streamlined ERP software testing with Selenium automation.')
    ]
    c.executemany('INSERT OR REPLACE INTO experience VALUES (?,?,?,?)', experiences)
    
    # Projects
    projects = [
        ('Personality Prediction System', 'Led a team to build a system for predicting employee personalities.', 
         'https://link.springer.com/chapter/10.1007/978-981-99-5354-7_15'),
        ('Ride Insights', 'Engineered data insights for NYC taxi trip records.', None)
    ]
    c.executemany('INSERT OR REPLACE INTO projects VALUES (?,?,?)', projects)
    
    # Education
    education = [
        ('Master of Science in Computer Science', 'San Jose State University', 'Aug 2023 – May 2025', '3.83/4.0'),
        ('Bachelor of Science in Computer Engineering', 'University of Mumbai', 'Aug 2017 - May 2021', '3.38/4.0')
    ]
    c.executemany('INSERT OR REPLACE INTO education VALUES (?,?,?,?)', education)
    
    conn.commit()

def get_skills(conn):
    c = conn.cursor()
    return c.execute('SELECT category, skill FROM skills').fetchall()

def get_experience(conn):
    c = conn.cursor()
    return c.execute('SELECT title, company, duration, description FROM experience').fetchall()

def get_projects(conn):
    c = conn.cursor()
    return c.execute('SELECT title, description, publication FROM projects').fetchall()

def get_education(conn):
    c = conn.cursor()
    return c.execute('SELECT degree, institution, duration, gpa FROM education').fetchall()

def ai_qa(question, content):
    """Use Google's Generative AI to answer questions based on the content."""
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an AI assistant for Mihir Dhirajlal Satra. Your task is to answer questions about Mihir's resume 
    based on the following content. If the answer is not in the content, respond with 
    "I'm sorry, I don't have that information in my current data."

    Resume Content:
    {content}

    Question: {question}

    Answer:
    """
    response = model.generate_content(prompt)
    return response.text



def get_pdf_text(file_path):
    """Extracts text from the provided PDF file."""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

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
    image = Image.open("Mihir.jpg")  
    st.sidebar.image(image, caption="Mihir Dhirajlal Satra", use_column_width=True)
   
def render_header():
    """Displays a header saying 'Hi, I am Mihir' at the top of every page."""
    st.markdown('<h1 style="text-align:center;">Hi, I am Mihir!</h1>', unsafe_allow_html=True)

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
    section = st.sidebar.selectbox("Select a section to view:", ("About", "Skills", "Work Experience", "Projects", "Education", "Gallery","Contact Me", "Download Resume"))

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


    st.header("💬 Ask Me Anything")
    user_question = st.text_input("You: ", placeholder="Type your question here...")
    if user_question:
        try:
            file_path = "MihirDhirajlal_Satra_Resume.pdf"
            resume_content = get_pdf_text(file_path)
            with st.spinner("Thinking..."):
                response = ai_qa(user_question, resume_content)
            st.write("Answer:", response)
        except FileNotFoundError:
            st.error(f"Error: The PDF file '{file_path}' was not found. Please make sure it's in the correct location.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.write("© 2024 Mihir Dhirajlal Satra's Resume Website | Powered by Streamlit")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()