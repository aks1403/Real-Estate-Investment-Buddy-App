import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.graphs import Neo4jGraph

# Set up environment variables and tools
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["FIRECRAWL_API_KEY"] = os.getenv('FIRECRAWL_API_KEY')
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["NEO4J_URI"] = os.getenv('NEO4J_URI')
os.environ["NEO4J_USERNAME"] = os.getenv('NEO4J_USERNAME')
os.environ["NEO4J_PASSWORD"] = os.getenv('NEO4J_PASSWORD')

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
search_tool = SerperDevTool()

# Define agents
researcher = Agent(
    llm=model,
    role="Senior Property Researcher",
    goal="Find promising investment properties.",
    backstory="You are a veteran property analyst with 20 years of experience. You specialize in identifying high-potential retail properties for investment.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

writer = Agent(
    llm=model,
    role="Senior Property Analyst",
    goal="Summarise property facts into a concise report for investors.",
    backstory="You are an experienced real estate agent with a talent for distilling complex property information into clear, actionable insights for potential investors.",
    allow_delegation=False,
    verbose=True,
)

# Define tasks
def create_tasks(city, country):
    task1 = Task(
        description=f"""
        Use the search tool to find 5 promising real estate investment suburbs in {city}, {country}. 
        For each suburb, provide a detailed report including:
        1. Mean, low, and maximum property prices
        2. Rental yield
        3. Recent price trends
        4. Population demographics
        5. Local amenities (schools, hospitals, shopping centers, public transport)
        6. Any recent or planned infrastructure developments
        7. Economic factors affecting the area
        8. Potential risks or challenges for investors

        Ensure that your report is comprehensive and includes specific data points and statistics where available.
        """,
        expected_output="""
        A detailed report of each of the 5 suburbs, containing all requested information.
        Background Information: These suburbs are typically located near major transport hubs, employment centers, and educational institutions. The report should highlight why each suburb is considered a top contender for investment opportunities.
        """,
        agent=researcher,
        output_file="task1_output.txt",
    )

    task2 = Task(
        description="""
        Review the detailed report provided by the researcher. For each of the 5 suburbs:
        1. Summarize the key points in 2-3 concise sentences.
        2. Highlight the most important features that would appeal to investors.
        3. Provide a brief assessment of the investment potential.

        Your summary should be clear, concise, and focused on the most relevant information for potential investors.
        """,
        expected_output="""
        A summarized list of the 5 suburbs, with each suburb's key points presented in 2-3 sentences, focusing on the most important features and investment potential.
        """,
        agent=writer,
        output_file="task2_output.txt",
    )

    return [task1, task2]

# Function to run the analysis
def run_analysis(city, country):
    tasks = create_tasks(city, country)

    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    return result

# Streamlit UI
st.title("Property Investment Analysis Tool")

st.write("Welcome to the Property Investment Analysis Tool. This application analyzes real estate markets and provides investment recommendations.")

city = st.text_input("Enter the city you want to analyze:")
country = st.text_input("Enter the country:")

if st.button("Run Analysis"):
    if city and country:
        with st.spinner(f"Analyzing property investment opportunities in {city}, {country}..."):
            result = run_analysis(city, country)
        
        st.success("Analysis complete!")
        st.subheader("Investment Report")
        st.text(result)
    else:
        st.warning("Please enter both city and country.")

# Chatbot integration
st.sidebar.header("Chat with the Bot")
user_query = st.sidebar.text_input("Ask a question about the property analysis:")

if user_query:
    # Opening the text file saved by agent 1
    with open('task1_output.txt', 'r') as file:
        content = file.read()

    embeddings = OpenAIEmbeddings()
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"]
    )

    llm_transformer = LLMGraphTransformer(llm=model)
    documents = [Document(page_content=content)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents=graph_documents, baseEntityLabel=True, include_source=True)

    chain = GraphCypherQAChain.from_llm(model, graph=graph, verbose=True)
    response = chain.invoke(user_query)

    st.sidebar.subheader("Chatbot Response")
    st.sidebar.text(response)

st.sidebar.header("About")
st.sidebar.info("This tool uses AI-powered agents to analyze real estate markets and provide investment recommendations.")

st.sidebar.header("How it works")
st.sidebar.markdown("""
1. Enter the city and country you're interested in.
2. Click 'Run Analysis' to start the process.
3. The tool will gather data, perform analysis, and generate a comprehensive report.
4. Use the chatbot to ask specific questions about the analysis.
5. Review the results to make informed investment decisions.
""")
