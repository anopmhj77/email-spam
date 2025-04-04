import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import nltk 
nltk.download('punkt')
import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from wordcloud import WordCloud



st.title("Data Visualization")

st.subheader("📂 Reading the Dataset")
df=pd.read_csv("spam.csv", encoding='ISO-8859-1')
df

st.subheader("📏 Shape of Dataset")
st.write(df.shape)

st.subheader("ℹ️ Info of Dataset")
st.write(df.info)

st.subheader("🗑️ droping nan colums")
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
st.write(df.sample(5))

st.subheader("✏️ Renaming the colums header")
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
st.write(df.sample(5))

st.subheader("🔢Giving value 0 for ham and 1 for spam")
df['target']=encoder.fit_transform(df['target'])
st.write(df['target'])
st.write(df.head(5))

st.subheader("⚠️Checking null and duplicated value")
st.write("✅ null value :")
st.write(df.isnull().sum())
st.write("✅duplicated value :")
st.write("before")
st.write(df.duplicated().sum())
df=df.drop_duplicates(keep='first')
st.write("after")
st.write(df.duplicated().sum())


st.subheader("📊 Spam vs Ham Distribution")
st.write(df['target'].value_counts())
fig = px.pie(df, names=df['target'].map({0: 'ham', 1: 'spam'}), title=" 📊Spam vs Ham Distribution")
st.plotly_chart(fig)


st.subheader("📏 Message Length Analysis")
df['num']=df['text'].apply(len)
st.write(df['num'])

st.subheader("🔠 Word Token Count in Messages")
df['token_count'] = df['text'].apply(lambda x: len(nltk.word_tokenize(str(x)))) 
st.write(df[['token_count']])


df['sent']=df['text'].apply(lambda x: len(nltk.sent_tokenize(str(x))))

st.subheader("📄 Sample Data (First 5 Rows)")
st.write(df.head(5))


st.write(df[df['target']==0][['num','token_count','sent']].describe())
st.write(df[df['target']==1][['num','token_count','sent']].describe())

sns.histplot(df[df['target'] == 0]['num'], kde=True)
sns.histplot(df[df['target'] == 1]['num'], kde=True,color='red')
st.pyplot() 

st.subheader("📊 Correlation Matrix (Numeric Columns Only)")
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
st.write(numeric_df.corr())  
st.subheader("🔍 Correlation Heatmap (Numeric Columns Only)")
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
st.pyplot() 




ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english')]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)  # Convert list back to a string

# Load dataset
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Apply text transformation
df['transformed_text'] = df['text'].apply(transform_text)

# Display dataframe
st.write(df.head(5))

st.subheader("🌥️ Word Cloud - Spam Messages")
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Create word cloud for spam messages
spam_text = df[df['target'] == 1]['transformed_text'].str.cat(sep=" ")
spam_wc = wc.generate(spam_text)

# Display the word cloud
plt.figure(figsize=(6,6))
plt.imshow(spam_wc)
plt.axis("off")  
st.pyplot()



ham_text = df[df['target'] == 0]['transformed_text'].str.cat(sep=" ")
ham_wc = wc.generate(ham_text)

# Display the word cloud
plt.figure(figsize=(6,6))
plt.imshow(spam_wc)
plt.axis("off")  
st.pyplot()


spam_corpus = []

for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)



from collections import Counter


st.subheader("📊 Top Words in Spam Messages")

spam_corpus = Counter(" ".join(df[df['target'] == 1]['transformed_text']).split())

spam_df = pd.DataFrame(spam_corpus.most_common(30), columns=['Word', 'Count'])


fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=spam_df, x='Word', y='Count', palette='viridis', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title("Top 30 Most Common Words in Spam Messages")
st.pyplot(fig)  
# Plotly Bar Chart


st.subheader("📊 Top Words in Ham Messages")
ham_corpus = Counter(" ".join(df[df['target'] == 0]['transformed_text']).split())

ham_df = pd.DataFrame(ham_corpus.most_common(30), columns=['Word', 'Count'])
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=ham_df, x='Word', y='Count', palette='viridis', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title("Top 30 Most Common Words in ham Messages")
st.pyplot(fig)  



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['transformed_text']).toarray()

# Display shape of transformed data
st.write("Feature Matrix Shape:", x.shape)

# Define target variable
y = df['target'].values
st.write("Target Variable:", y)

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize Naive Bayes classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# --- GaussianNB ---
gnb.fit(x_train, y_train)
y_pred1 = gnb.predict(x_test)
st.write("📊 **Gaussian Naive Bayes**")
st.write("Accuracy:", accuracy_score(y_test, y_pred1))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
st.write("Precision Score:", precision_score(y_test, y_pred1))

# --- MultinomialNB ---
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
st.write("📊 **Multinomial Naive Bayes**")
st.write("Accuracy:", accuracy_score(y_test, y_pred2))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
st.write("Precision Score:", precision_score(y_test, y_pred2))

# --- BernoulliNB ---
bnb.fit(x_train, y_train)
y_pred3 = bnb.predict(x_test)
st.write("📊 **Bernoulli Naive Bayes**")
st.write("Accuracy:", accuracy_score(y_test, y_pred3))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))
st.write("Precision Score:", precision_score(y_test, y_pred3))



import pickle 

pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))