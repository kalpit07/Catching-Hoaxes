from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import load_model
import pickle


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Loading Saved Model
    tfidf_vectorizer = pickle.load(open("pklFiles/tfidf_vectorizer.pkl", "rb"))
    svm_classifier = pickle.load(
        open("pklFiles/svm_classifier_for_tfidf_vectorizer.pkl", "rb"))
    cnn_model = load_model('Trained Models/cnn_model.h5')


    if request.method == 'POST':
        user_headline = request.form['user_headline']
        news_author = request.form['user_headline_author']
        news = request.form['user_news']
        user_headline_list = [user_headline]
        headline_count = tfidf_vectorizer.transform(user_headline_list)
        prediction = svm_classifier.predict(headline_count)

        def predict_news_cnn(title, author, text):
            # Lower case
            total_info = title + ' ' + author + ' ' + text
            total_info = total_info.lower()
    
            # Removing punctuations
            def Punctuation(string):
                punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
                for x in string.lower():
                    if x in punctuations:
                        string = string.replace(x, "")
        
                return string
    
            total_info = Punctuation(total_info)
    
            # Eliminating extra spaces
            total_info = total_info.replace('   ', ' ')
            total_info = total_info.replace('  ', ' ')
    
            # Removing stopwords
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
    
            stop_words = set(stopwords.words('english')) 
            word_tokens = word_tokenize(total_info) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            total_info = ''
            for word in filtered_sentence:
                total_info += word + ' '
        
            total_info = total_info.rstrip()
    
            # Loading tokenizer
            tokenizer = pickle.load(open('pklFiles/tokenizer.pkl', 'rb'))
    
            # Defining variables for maxlen
            MAX_SEQUENCE_LENGTH = 500
    
            test_sequence = tokenizer.texts_to_sequences([total_info])
    
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            test_data = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
            # Loading CNN model
            from tensorflow.keras.models import load_model
            cnn_model = load_model('Trained Models/cnn_model.h5')
    
            # Prediction
            predicted_label = cnn_model.predict(test_data)
    
            # Result
            if int(predicted_label.round()[0][0]) == 1:
                return "Reliable"
            else:
                return "Unreliable"

        reliability = predict_news_cnn(user_headline, news_author, news)



        return render_template('Result.html', headline=user_headline_list[0], prediction=prediction, reliability=reliability)


if __name__ == '__main__':
    app.run(debug=True)