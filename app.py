from flask import Flask,render_template,url_for,request
import pickle
import joblib

clf = pickle.load(open('SpamDetectionModel', 'rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
      return render_template('home.html')
      
@app.route('/predict',methods=['POST'])
def predict():
    if request .method == 'POST':
        message = request.form['message']
        message = open('test_gmail/mails.txt','r').read().split('\n')[0]
        data = [message]
        vect=cv.transform(data)
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host="localhost", port=8002, debug=True)


