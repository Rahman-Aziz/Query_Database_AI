# ==========================================
# 1. IMPORTING REQUIRED LIBRARIES
# ==========================================
import streamlit as st 
import pandas as pd
from langchain_community.utilities import SQLDatabase 
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq 
from langchain_community.chat_models import ChatOllama 
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 2. CACHING FUNCTIONS FOR SPEED
# ==========================================

@st.cache_resource(show_spinner="Connecting to Database...")
def get_database_connection(uri):
    return SQLDatabase.from_uri(
        uri, 
        sample_rows_in_table_info=0 
    )

@st.cache_resource(show_spinner="Waking up the AI Brain...")
def get_llm(ai_choice, api_key):
    if ai_choice == "OpenAI (Paid)":
        return ChatOpenAI(api_key=api_key, temperature=0, model_name="gpt-4o")
    elif ai_choice == "Groq (Free Cloud - Recommended)":
        # FIXED: Changed from 'groq/compound' to a real model that supports tool calling perfectly
        return ChatGroq(groq_api_key=api_key, temperature=0, model_name="llama-3.3-70b-versatile")
    elif ai_choice == "Ollama (100% Free Local)":
        # FIXED: Ensure this matches the exact model name you pulled in Ollama (e.g., llama3, llama3.2, or mistral)
        return ChatOllama(model="llama3", temperature=0)
    return None

# ==========================================
# 3. SETTING UP THE USER INTERFACE (UI)
# ==========================================
st.title("🤖 Multi-Brain Database Analyst")

ai_choice = st.selectbox(
    "Choose your AI Brain:", 
    ["Groq (Free Cloud - Recommended)", "Ollama (100% Free Local)", "OpenAI (Paid)"]
)

api_key = ""
if ai_choice == "OpenAI (Paid)":
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
elif ai_choice == "Groq (Free Cloud - Recommended)":
    api_key = st.text_input("Enter your Groq API Key:", type="password")
else:
    st.info("Ollama selected! No API key needed. (Make sure the Ollama app is running on your PC)")

db_uri = st.text_input("Enter Database Connection URI:", placeholder="sqlite:///sample.db")
user_query = st.text_area("What would you like to know about the data?")

if st.button("Analyze Data"):
    
    if (ai_choice != "Ollama (100% Free Local)" and not api_key) or not db_uri or not user_query:
        st.warning("Please provide all required fields.")
    else:
        with st.spinner(f"Analyzing data using {ai_choice}..."):
            try:
                # 1. Fetch cached database
                db = get_database_connection(db_uri)
                
                # 2. Fetch cached LLM
                llm = get_llm(ai_choice, api_key)

                # ==========================================
                # 4. EXECUTING THE AGENT LOOP
                # ==========================================
                
                # Dynamically choose the best configuration pattern based on providers
                if ai_choice == "OpenAI (Paid)":
                    current_agent_type = "openai-tools"
                else:
                    current_agent_type = "tool-calling" # Universal standard for Groq/Ollama

                agent_executor = create_sql_agent(
                    llm=llm,
                    db=db,
                    agent_type=current_agent_type,  
                    verbose=True,     
                    return_intermediate_steps=True            
                )
                
                # CRITICAL SYSTEM INSTRUCTION: Forces the AI to output paragraphs as structured Markdown grids
                system_instruction = "\n\nCRITICAL FORMATTING RULES: If your answer includes columns, data rows, or query results from tables, you MUST format them as a clean, standardized Markdown Table syntax. Do not print rows as plain conversational sentences."
                
                # Execute the agent query safely
                response = agent_executor.invoke({"input": user_query + system_instruction})
                
                # ==========================================
                # 5. DISPLAYING THE RESULTS
                # ==========================================
                st.success("Analysis Complete!")
                
                # Setup UI tabs
                tab1, tab2, tab3 = st.tabs(["Interactive Data Grid", "AI Summary", "Executed SQL Code"])
                
                # Attempt to dynamically extract raw table data from the agent's internal thought logs
                raw_data_extracted = None
                executed_sql_string = "No query recorded."
                
                if "intermediate_steps" in response:
                    for action, observation in response["intermediate_steps"]:
                        # Identify when the agent called the sql_db_query tool
                        if hasattr(action, 'tool') and action.tool == "sql_db_query":
                            executed_sql_string = action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)
                            # Convert the raw database string output into a clean list/dict structure
                            try:
                                if observation and "Error" not in observation:
                                    # Use Langchain's built-in DB utility to fetch records directly for our UI grid
                                    raw_data_extracted = db.run(executed_sql_string, fetch="cursor").fetchall()
                                    column_names = db.run(executed_sql_string, fetch="cursor").keys()
                            except Exception:
                                pass # Fallback to AI markdown text if database parsing hiccuped

                with tab1:
                    st.subheader("Data Grid View")
                    if raw_data_extracted and column_names:
                        # Dynamically load data into a modern DataFrame with zero hardcoding
                        df = pd.DataFrame(raw_data_extracted, columns=column_names)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No raw database dataset returned for this specific request, or the query didn't return tabular rows.")
                
                with tab2:
                    st.subheader("AI Narrative Report")
                    # Renders the clean markdown table generated by the new system prompt instruction
                    st.markdown(response["output"])
                    
                with tab3:
                    st.subheader("Underlying Database Query Run")
                    st.code(executed_sql_string, language="sql")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
