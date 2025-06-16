import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import google.generativeai as genai

# --- Configuration ---
# Set page configuration for Streamlit
st.set_page_config(layout="wide", page_title="AI-Powered Data Analyst")

# --- Initialize Gemini API Key ---
# Instructions state to leave apiKey as an empty string and Canvas will provide it
# const apiKey = ""
# If you are running this outside Canvas, you'll need to replace "" with your actual API key
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "") # Example for Streamlit Cloud secrets
genai.configure(api_key="") # Leave empty for Canvas environment

# --- Helper Function to Call Gemini API ---
def get_gemini_response(prompt_parts):
    """Sends a prompt to the Gemini model and returns the text response."""
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        # Ensure prompt_parts is a list of dictionaries if multimodal input is intended,
        # otherwise, just a string is fine for text-only.
        # For simplicity with text-only prompts from context, we ensure it's a list if it's not already.
        if not isinstance(prompt_parts, list):
            prompt_parts = [{"text": prompt_parts}]

        response = model.generate_content(prompt_parts)
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "No response generated or unexpected response structure from AI."
    except Exception as e:
        st.error(f"Error communicating with AI: {e}")
        return "An error occurred while getting a response from the AI."

# --- Streamlit App UI ---
st.title("📊 AI-Powered Data Analyst")
st.markdown("Upload your CSV, explore insights, and chat with your data!")

# --- Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data into DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df # Store in session state

        st.success("CSV file uploaded successfully!")

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # --- Data Preprocessing Options ---
        st.subheader("Data Preprocessing")
        preprocessing_option = st.selectbox(
            "Select preprocessing for missing values:",
            ["None", "Drop rows with any missing values", "Fill numerical with mean", "Fill categorical with mode"]
        )

        processed_df = df.copy()
        if preprocessing_option == "Drop rows with any missing values":
            processed_df.dropna(inplace=True)
            st.info("Missing values rows dropped.")
        elif preprocessing_option == "Fill numerical with mean":
            for col in processed_df.select_dtypes(include=['number']).columns:
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            st.info("Numerical missing values filled with column mean.")
        elif preprocessing_option == "Fill categorical with mode":
            for col in processed_df.select_dtypes(include=['object', 'category']).columns:
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
            st.info("Categorical missing values filled with column mode.")

        st.session_state.df = processed_df # Update df in session state

        st.subheader("Preprocessed Data Info")
        buffer = io.StringIO()
        processed_df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Descriptive Statistics")
        st.dataframe(processed_df.describe())

        # --- Generate Summary with AI ---
        st.subheader("AI-Generated Data Summary")
        if st.button("Generate Summary (AI)", key="generate_summary_btn") or st.session_state.summary_generated:
            st.session_state.summary_generated = True
            with st.spinner("Analyzing data and generating summary..."):
                # Prepare data for LLM
                df_info_str = processed_df.info(buf=io.StringIO()) # Capture info to string
                df_describe_str = processed_df.describe().to_string()
                column_names = processed_df.columns.tolist()
                categorical_cols_sample = {}
                for col in processed_df.select_dtypes(include=['object', 'category']).columns:
                    categorical_cols_sample[col] = processed_df[col].value_counts().head(5).to_dict()

                prompt_summary = f"""
                You are an AI data analyst. I will provide you with information about a CSV dataset.
                Your task is to analyze the data and provide a concise summary highlighting key trends, patterns, and insights.
                Focus on:
                1.  **Data Overview:** Number of rows/columns, missing values (if any significant), data types.
                2.  **Key Statistics:** Important insights from numerical descriptive statistics (mean, median, std, min, max).
                3.  **Categorical Insights:** Dominant categories, unique values.
                4.  **Potential Relationships/Trends:** Hypothesize about relationships between columns.

                Here is the data information:
                DataFrame Info:
                {buffer.getvalue()}

                Descriptive Statistics (Numerical Columns):
                {df_describe_str}

                Column Names: {column_names}

                Top 5 Categories for Categorical Columns:
                {json.dumps(categorical_cols_sample, indent=2)}

                Please provide a summary of trends, patterns, and insights from this data.
                """
                summary_text = get_gemini_response(prompt_summary)
                st.write(summary_text)

        # --- Data Visualizations ---
        st.subheader("Data Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Histograms / Bar Plots")
            selected_column_hist = st.selectbox(
                "Select a column for Histogram/Bar Plot:",
                processed_df.columns
            )
            if selected_column_hist:
                fig, ax = plt.subplots(figsize=(10, 6))
                if pd.api.types.is_numeric_dtype(processed_df[selected_column_hist]):
                    sns.histplot(processed_df[selected_column_hist], kde=True, ax=ax)
                    ax.set_title(f"Distribution of {selected_column_hist}")
                    ax.set_xlabel(selected_column_hist)
                    ax.set_ylabel("Frequency")
                else:
                    # Count plot for categorical data
                    sns.countplot(y=processed_df[selected_column_hist], order=processed_df[selected_column_hist].value_counts().index, ax=ax)
                    ax.set_title(f"Count of {selected_column_hist}")
                    ax.set_xlabel("Count")
                    ax.set_ylabel(selected_column_hist)
                st.pyplot(fig)

        with col2:
            st.markdown("#### Scatter Plot / Pair Plot")
            numerical_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                x_col = st.selectbox("Select X-axis for Scatter Plot:", numerical_cols, index=0)
                y_col = st.selectbox("Select Y-axis for Scatter Plot:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)

                if x_col and y_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=processed_df[x_col], y=processed_df[y_col], ax=ax)
                    ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
            else:
                st.info("Need at least two numerical columns for scatter plot.")

        st.markdown("#### Correlation Heatmap")
        numerical_df = processed_df.select_dtypes(include=['number'])
        if not numerical_df.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap of Numerical Columns")
            st.pyplot(fig)
        else:
            st.info("No numerical columns to display correlation heatmap.")

        # --- Chat with the File ---
        st.subheader("Chat with Your Data (AI)")
        st.markdown("Ask questions about your data (e.g., 'What is the average age?', 'Show me the top 5 customers by sales?', 'What are the most common categories in column X?').")

        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input from user
        user_prompt = st.chat_input("Ask a question about your data...")

        if user_prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Prepare context for the AI
            # Sending descriptive stats and info to LLM for context
            df_snapshot_for_ai = {
                "info": buffer.getvalue(),
                "describe": processed_df.describe().to_string(),
                "columns": processed_df.columns.tolist(),
                "head": processed_df.head().to_string(),
            }
            # Add a small sample of the dataframe for more specific queries
            # For very large dataframes, you might need more sophisticated sampling or retrieval augmented generation (RAG)
            data_sample_str = processed_df.sample(min(5, len(processed_df))).to_string() if len(processed_df) > 0 else "DataFrame is empty."

            llm_context_prompt = f"""
            You are an AI data analyst. You have access to a CSV file.
            Here is some information about the DataFrame:

            DataFrame Info:
            {df_snapshot_for_ai['info']}

            Descriptive Statistics (Numerical Columns):
            {df_snapshot_for_ai['describe']}

            Column Names: {df_snapshot_for_ai['columns']}

            Sample Data Rows:
            {data_sample_str}

            Based on this information and assuming you have access to the full dataset in memory for detailed analysis, answer the following question.
            If the question implies a calculation or specific data retrieval, assume you can perform it on the full dataset and provide the result/insight.

            User's question: {user_prompt}
            """

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_response = get_gemini_response(llm_context_prompt)
                    st.markdown(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"Error processing file or data: {e}. Please ensure it's a valid CSV.")
        st.session_state.df = None # Reset dataframe on error
        st.session_state.chat_history = []
        st.session_state.summary_generated = False
else:
    st.info("Please upload a CSV file to begin your data analysis.")

