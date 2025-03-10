import streamlit as st
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classification_agent(exception_details):
    prompt = (
        "You are an expert in fund administration exception management. "
        "Based on the following exception details, classify the exception by type, priority, and complexity:\n\n"
        f"{exception_details}\n\n"
        "Provide your answer in a JSON format with keys 'type', 'priority', and 'complexity'."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a classification expert. Always return valid JSON."}, 
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        content = response.choices[0].message.content
        # Ensure the response is valid JSON
        json.loads(content)  # Validate JSON format
        return content
    except json.JSONDecodeError as e:
        return json.dumps({
            "type": "Error: Invalid response format",
            "priority": "N/A",
            "complexity": f"Failed to process the classification: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "type": "Error: System error",
            "priority": "N/A",
            "complexity": f"An unexpected error occurred: {str(e)}"
        })

def resolution_suggestion_agent(exception_details):
    prompt = (
        "You are a fund administration expert. Given the following exception details and historical resolution patterns, "
        "suggest a corrective action. Include a confidence score (as a percentage string) and rationale for your recommendation.\n\n"
        f"{exception_details}\n\n"
        "Present your answer in a valid JSON format with the following structure:\n"
        "{'suggestion': 'your suggestion here', 'confidence': '85%', 'rationale': 'your rationale here'}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a resolution suggestion expert. Always return valid JSON."}, 
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        content = response.choices[0].message.content
        # Ensure the response is valid JSON
        json.loads(content)  # Validate JSON format
        return content
    except json.JSONDecodeError as e:
        return json.dumps({
            "suggestion": "Error: Invalid response format",
            "confidence": "0%",
            "rationale": f"Failed to process the suggestion: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "suggestion": "Error: System error",
            "confidence": "0%",
            "rationale": f"An unexpected error occurred: {str(e)}"
        })

def explanation_agent(suggestion_details):
    prompt = (
        "Explain the following resolution suggestion in clear, concise natural language so that an operator can easily understand it:\n\n"
        f"{suggestion_details}"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an explanation expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

def process_exception(exception_details):
    try:
        agents = {
            "classification": classification_agent,
            "resolution": resolution_suggestion_agent,
            "explanation": explanation_agent
        }
        results = {}
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            future_to_agent = {executor.submit(func, exception_details): name for name, func in agents.items()}
            for future in future_to_agent:
                agent_name = future_to_agent[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    st.error(f"Error in {agent_name}: {str(e)}")
                    results[agent_name] = f"Error: {e}"
        return results
    except Exception as e:
        st.error(f"Main process error: {str(e)}")
        return {}

def main():
    # Set page config for a wider layout
    st.set_page_config(layout="wide", page_title="AI-Powered Fund Administration Exception Management")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {padding: 2rem;}
        .stButton>button {width: 100%;}
        .stats-box {padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin: 0.5rem;}
        </style>
    """, unsafe_allow_html=True)

    # Header with company branding
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 AI-Powered Fund Administration Exception Management")
        st.markdown("""<p style='font-size: 1.2em; color: #666;'>
            Revolutionizing exception handling with advanced AI technology</p>""", 
            unsafe_allow_html=True)

    # Dashboard metrics (simulated)
    st.markdown("### 📊 System Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='stats-box'>
            <h3 style='text-align: center; color: #0066cc;'>98%</h3>
            <p style='text-align: center;'>Resolution Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='stats-box'>
            <h3 style='text-align: center; color: #0066cc;'>85%</h3>
            <p style='text-align: center;'>Automation Rate</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='stats-box'>
            <h3 style='text-align: center; color: #0066cc;'>60s</h3>
            <p style='text-align: center;'>Avg. Resolution Time</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='stats-box'>
            <h3 style='text-align: center; color: #0066cc;'>24/7</h3>
            <p style='text-align: center;'>System Availability</p></div>""", unsafe_allow_html=True)

    # Main content area
    st.markdown("### 🔄 Exception Processing Engine")
    col1, col2 = st.columns([2, 1], gap="large")

    # Initialize results variable
    results = {}

    with col1:
        st.container()
        exception_details = st.text_area(
            "Enter Exception Details",
            height=150,
            placeholder="Paste exception details or sample log here...")

        if st.button("🚀 Process Exception", use_container_width=True) and exception_details:
            with st.spinner("🔍 AI Engine Processing..."):
                results = process_exception(exception_details)
            st.success("✅ Analysis Complete!")

            # Results container with two columns
            st.markdown("<div style='min-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            
            # Left column: Classification and Resolution
            with col_left:
                try:
                    classification = json.loads(results.get("classification", "{}"))
                    st.markdown("#### Classification Details")
                    if classification:
                        st.markdown(f"**Type:** {classification.get('type', 'N/A')}")
                        st.markdown(f"**Priority:** {classification.get('priority', 'N/A')}")
                        st.markdown(f"**Complexity:** {classification.get('complexity', 'N/A')}")
                    else:
                        st.error("No classification data available")
                except Exception as e:
                    st.error(f"Classification processing error: {str(e)}")
                    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

            # Right column: Resolution and Explanation
            with col_right:
                try:
                    resolution = json.loads(results.get("resolution", "{}"))
                    st.markdown("#### AI-Generated Resolution")
                    st.markdown(f"**Suggestion:** {resolution.get('suggestion', 'N/A')}")
                    st.progress(float(resolution.get('confidence', '0').rstrip('%'))/100)
                    st.markdown(f"**Confidence Score:** {resolution.get('confidence', 'N/A')}")
                    st.info(f"**Rationale:** {resolution.get('rationale', 'N/A')}")
                    
                    st.markdown("#### Detailed Explanation")
                    st.info(results.get("explanation", "No explanation available."))
                except Exception:
                    st.error("Unable to process resolution suggestion")

    with col2:
        # Only show Action Center and Insights after processing
        if exception_details and results:
            st.markdown("### 🎯 Action Center")
            st.markdown("""<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;'>
                <h4>Resolution Workflow</h4></div>""", unsafe_allow_html=True)
            
            # Action buttons
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("✅ Approve", key="approve_btn"):
                    st.success("Resolution approved for execution")
            with col_reject:
                if st.button("❌ Reject", key="reject_btn"):
                    st.error("Resolution rejected")
            
            if st.button("✏️ Modify", use_container_width=True, key="modify_btn"):
                st.info("Opening modification interface...")

            # Additional insights
            st.markdown("### 📈 Insights")
            st.markdown("""<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;'>
                <p><strong>AI Confidence:</strong> High</p>
                <p><strong>Similar Cases:</strong> 15 found</p>
                <p><strong>Avg. Resolution Time:</strong> 45 seconds</p>
                </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()