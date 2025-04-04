import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Multpage Spam Detection App",
    page_icon="📩",
)



# Title with emoji
st.title("About Project 🤔📊")
st.subheader("📌 Overview")
st.write("This project focuses on building a **machine learning model** to classify SMS messages as **spam 🚨** or **ham ✅ (not spam)** using the **UCI ML SMS Spam Collection Dataset**. The dataset contains **5,574** labeled messages, making it a great resource for **Natural Language Processing (NLP)** and text classification tasks.")

# Objective section with bullet emojis
st.subheader("🎯 Objective")
st.write("""
The main goal is to develop a **spam detection model** that can automatically identify spam messages based on text features.  
It can be used in applications like:

🔹📱 **SMS filtering** for mobile networks  
🔹📧 **Email spam detection**  
🔹💬 **Chat moderation systems**  
""")

# Dataset description
st.subheader("📊 Dataset Description")
st.write("""
The dataset consists of two columns:

🔹🟢 **Label**: Indicates whether a message is **spam 🚨** or **ham ✅ (not spam)**  
🔹✉️ **Message**: The actual SMS text  
""")

# Project steps
st.subheader("🔄 Project Steps")

st.subheader("🛠 Step 1: Data Preprocessing")
st.write("""
🔹 Load the dataset using **Pandas**  
🔹 Convert labels (**ham ✅** and **spam 🚨**) into numerical format (**0 = ham, 1 = spam**)  
🔹 **Text Cleaning**: Remove special characters, numbers, and stopwords  
🔹 **Tokenization**: Split messages into words  
🔹 **Lemmatization**: Reduce words to their base form  
""")

st.subheader("📌 Step 2: Feature Extraction")
st.write("""
Since the dataset contains **text data**, we convert it into numerical form using:

🔹📊 **TF-IDF (Term Frequency-Inverse Document Frequency)**  
🔹📦 **Bag of Words (BoW)**  
🔹🧠 **Word Embeddings** *(optional, for deep learning models)*  
""")

st.subheader("🤖 Step 3: Model Selection & Training")
st.write("""
 We train different **machine learning models**, including:

🔹⚡ **Naïve Bayes** (Common for text classification)  
🔹📈 **Logistic Regression**  
🔹🌲 **Random Forest**  
🔹💡 **Support Vector Machine (SVM)**  
🔹🧠 **Deep Learning** *(LSTMs or Transformers like BERT, optional)*  
""")

st.subheader("📊 Step 4: Model Evaluation")
st.write("""
We evaluate models using:

🔹✅ **Accuracy, Precision, Recall, and F1-score**  
🔹📊 **Confusion Matrix** to see false positives/negatives  
🔹📈 **ROC Curve** to check classification performance  
""")

# Expected results
st.subheader("📌 Expected Results")
st.write("""
🔹🎯 A trained model that can classify SMS messages as **spam 🚨 or ham ✅**  
🔹📑 A well-documented **Jupyter Notebook** with data visualization and model comparisons  
""")

# Tools and libraries
st.subheader("🛠 Tools & Libraries")
st.write("""
🔹🐍 **Python** (Programming language)  
🔹📊 **Pandas, NumPy** (Data handling)  
🔹📖 **NLTK, spaCy** (Text processing)  
🔹🤖 **Scikit-learn** (Machine learning models)  
🔹📈 **Matplotlib, Seaborn** (Visualization)  
""")

# Future enhancements
st.subheader("🚀 Future Enhancements")
st.write("""
🔹🧠 Use **Deep Learning models** (LSTMs, Transformers like BERT)  
🔹📲 Implement **real-time SMS filtering** in applications  
🔹⚖️ Improve handling of **imbalanced datasets**  
 """)

# Conclusion
st.subheader("🎯 Conclusion")
st.write("This project demonstrates how to build a **spam detection system** using **machine learning and NLP techniques**. It is a great way to learn about **text classification, feature extraction, and model evaluation**. 🚀")

# Sidebar message
st.sidebar.success("🔍 Select any page above to explore! 😃✌️")
