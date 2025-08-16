import torch
import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "thebugged/Bert"
BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 512

# sample texts for testing
SAMPLE_TEXTS = [
    "Great job on this project!",
    "I disagree with your approach",
    "Thanks for sharing this",
    "This is completely wrong",
    "You're an idiot",
    "Nice work everyone",
    "What a stupid idea",
    "Go to hell!",
    "I appreciate your help",
    "You don't know anything"
]

@st.cache_resource
def load_model_and_tokenizer():
    """Load the pre-trained model and tokenizer"""
    with st.spinner("Loading model..."):
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval() 
    return tokenizer, model

def predict_toxicity(text, tokenizer, model):
    """Predict if text is toxic or not"""
    inputs = tokenizer(
        [text], 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = probabilities.detach().numpy()[0]
    
    predicted_class = np.argmax(outputs.logits.detach().numpy(), axis=1)[0]
    
    return predicted_class, probabilities, outputs.logits.detach().numpy()[0]

def main():
    st.set_page_config(
        page_title="TCC",
        page_icon="💭",
        layout="centered"
    )
    
    st.image("images/banner.png") 
    
    st.markdown("""
    Analyze text for toxic content using a fine-tuned BERT model. 
    Enter your own text or try the sample comments below.
    """)
    
    tokenizer, model = load_model_and_tokenizer()

    selected_sample = st.pills(
            "Quick samples",
            SAMPLE_TEXTS,
            selection_mode="single"
        )

    
    with st.container():
  
      
        if selected_sample:
            default_text = selected_sample
        else:
            default_text = st.session_state.get('current_text', '')
        
        user_input = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            placeholder="Type your comment here...",
            height=100,
            label_visibility="collapsed"
        )
        
   
        st.session_state.current_text = user_input

        analyze_button = st.button("Analyze Text", type="primary", use_container_width=False)
        
    
    # Results section
    if user_input.strip() and analyze_button:
      
        with st.spinner("Analyzing..."):
            predicted_class, probabilities, raw_logits = predict_toxicity(user_input, tokenizer, model)
    
        # probability bars
        non_toxic_prob = probabilities[0] * 100
        toxic_prob = probabilities[1] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Non-Toxic Probability", f"{non_toxic_prob:.1f}%")
            st.progress(non_toxic_prob / 100)
        
        with col2:
            st.metric("Toxic Probability", f"{toxic_prob:.1f}%")
            st.progress(toxic_prob / 100)
    
    elif analyze_button and not user_input.strip():
        st.warning("Please enter some text to analyze!")
  

if __name__ == "__main__":
    main()