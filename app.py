from flask import Flask  , request , render_template
import numpy as np
app = Flask(__name__)
import pickle
model=pickle.load(open("new_model.pkl" , "rb"))
pipeline=pickle.load(open("new_transformer.pkl" , "rb"))


@app.route('/')
def hello_world():
    return render_template("index2.html")

@app.route("/predict" , methods=["POST"])
def predict_approval():
    gender=request.form.get('Gender')
    married=request.form.get("Marry")
    depen=int(request.form.get('Dependents'))
    education=request.form.get('Educations')
    self=request.form.get('Self')
    appli=int(request.form.get('Applicant'))
    coapplicant=int(request.form.get('income'))
    loan=int(request.form.get('Loan'))
    loanterm=int(request.form.get('Term'))
    area=request.form.get('Area')
    credit=request.form.get("credit")
    
    vector=pipeline.transform([[gender ,married,depen,education ,self , appli ,coapplicant,loan , loanterm,credit ,area]])
    #vector=pipeline.transform(vector)
    #a=pipeline.transform([["Male" ,"No",0,"Graduate" ,"No" ,5849 ,0.0 ,135.0 , 360.0 ,1.0 , "Urban"]])
    final=model.predict(vector)[0]
    #lis=[gender , married , depen , education ,self , appli , coapplicant, loan ,loanterm ,  area , credit]
    final=1
    if final==1:
        result="Loan has Succesfully Approved"
    else:
        result="Sorry , Your Loan Application Has Been Reject"  
    return render_template("index2.html" , result=result)

if __name__ == '__main__':
    app.run(debug=True)