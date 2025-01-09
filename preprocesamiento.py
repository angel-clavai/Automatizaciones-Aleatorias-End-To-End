import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # minúsculas
    text = text.lower()
    
    # Eliminar caracteres no alfanuméricos
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenización
    words = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return ' '.join(lemmatized_words)


if __name__ == '__main__':
    preprocessed_text = preprocess_text(transcribed_text)
    print("Texto preprocesado:", preprocessed_text)
