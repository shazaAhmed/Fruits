
from flask import Flask,render_template,request,Response, url_for
from restools import *
from werkzeug.utils import secure_filename
import json
import os
app = Flask(__name__)


model= ResNetCNN()
model.load_state_dict(torch.load('fruits-360-resnet.pth',map_location=torch.device('cpu') ))
model.eval()


@app.route('/', methods=['GET'])
def choix():
    try:
        return render_template('prediction.html')
    except:
        return render_template('choix_user.html', mess="Reformuler votre choix")





@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        _, _, filenames = next(os.walk("static/subdata"))
        for file in filenames:
            os.remove("static/subdata/" + file)
        
        f = request.files['file']
        file_name=secure_filename(f.filename)
        
        path = "static/subdata/" + secure_filename(f.filename)
        f.save(path)
        
        test_tfms = tt.Compose([tt.ToTensor()])
        test_ds = ImageFolder("static", test_tfms)
        
        preds = []
        nutritions = []
        fruits_classes = pickle.load(open("fruits-classes.pkl", "rb"))
        for img in test_ds:
            pred = predict_image(img[0], model)
            preds.append(pred)
            valeur_nutritionnelle = get_nutritional_value_2(fruits_classes[pred])
            nutritions.append(valeur_nutritionnelle)
        
        return render_template('resultat.html', path = "subdata/"+secure_filename(f.filename),
                               fruits_classes=fruits_classes,
                               preds=preds, nutritions = nutritions)



if __name__ == "__main__":
    app.run(port="5001",debug=True)
    