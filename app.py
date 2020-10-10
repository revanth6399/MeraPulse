from flask import Flask, render_template, url_for, request 
import numpy as np 
import pickle 
app = Flask(__name__, static_folder='static',template_folder='templates')

@app.route("/") 
def home():     
  return render_template('home.html')  
 
@app.route('/result1', methods=['POST', 'GET']) 
def result1():     
  age = int(request.form['age'])     
  gender = int(request.form['gender'])     
  height = int(request.form['height'])     
  weight = int(request.form['weight'])     
  ap_hi = int(request.form['ap_hi'])     
  ap_lo = int(request.form['ap_lo'])     
  cholesterol = int(request.form['cholesterol'])     
  gluc = int(request.form['gluc'])     
  smoke = int(request.form['smoke']) 
  alco = int(request.form['alco'])  
  active = int(request.form['active'])        
  a = np.array([age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]).reshape(1, -1)
  print(a)

  model = pickle.load(open('heartuci.pkl','rb'))
  l = model.predict(a)   
  if l == 0:         
     return render_template('0.html')    
  else:         
     return render_template('1.html')    
 
@app.route('/result2', methods=['POST', 'GET']) 
def result2():     
  age = int(request.form['age'])     
  sex = int(request.form['sex'])     
  trestbps = float(request.form['trestbps'])     
  chol = float(request.form['chol'])     
  restecg = float(request.form['restecg'])     
  thalach = float(request.form['thalach'])     
  exang = int(request.form['exang'])     
  cp = int(request.form['cp'])     
  fbs = float(request.form['fbs']) 
  oldpeak = float(request.form['oldpeak'])  
  slope = int(request.form['slope'])  
  ca = int(request.form['ca'])  
  thal = int(request.form['thal'])      
  y = np.array([age, sex, cp, trestbps, chol, fbs, restecg,                   
           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
  print(y)

  
  model = pickle.load(open('cleveland.pkl','rb'))
  r = model.predict(y)   
  if r == 0:         
     return render_template('0.html')    
  else:         
     return render_template('1.html')  
 
@app.route("/aboutus/") 
def aboutus():     
    return render_template('aboutus.html') 
  
@app.route("/diseaseinfo/") 
def diseaseinfo():     
    return render_template('diseaseinfo.html') 

@app.route("/commonman/") 
def commonman():     
    return render_template('commonman.html') 

@app.route("/labtech/") 
def labtech():     
    return render_template('labtech.html') 



if __name__ == "__main__":     
     app.run(debug=True) 
     
     
