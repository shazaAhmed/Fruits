
from flask import Flask,render_template,request,Response, url_for
from restools import *
from werkzeug.utils import secure_filename
import json
#import conf
import os
app = Flask(__name__)

#user = conf.user
#path = conf.path

model= ResNetCNN()
model.load_state_dict(torch.load('fruits-360-resnet.pth',map_location=torch.device('cpu') ))
model.eval()

#@app.route('/')
#def use():
    #print(model)
    
    #image = Image.open("data/subdata/kmq.png")
    #img = ToTensor()
    #print(type(test_ds[0][0])) 
    #print(test_ds[0][0].unsqueeze(0))
    
    #print(preds)
    #print(fruits_classes[preds])
    #return render_template('choix_user.html')

@app.route('/', methods=['GET'])
def choix():
    try:
        return render_template('prediction.html')
    except:
        return render_template('choix_user.html', mess="Reformuler votre choix")





#model = TheModelClass(*args, **kwargs)
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        _, _, filenames = next(os.walk("static/subdata"))
        for file in filenames:
            os.remove("static/subdata/" + file)
        
        f = request.files['file']
        file_name=secure_filename(f.filename)
        #print(type(f))
        path = "static/subdata/" + secure_filename(f.filename)
        f.save(path)
        #path ="data/"+file_name
        test_tfms = tt.Compose([tt.ToTensor()])
        test_ds = ImageFolder("static", test_tfms)
        #predict_image( f, model)
        preds = []
        fruits_classes = pickle.load(open("fruits-classes.pkl", "rb"))
        for img in test_ds:
            preds.append(predict_image(img[0], model))
        #model.load_state_dict(torch.load(fruits-360-resnet))
        #model.eval()
        return render_template('resultat.html', path = "subdata/"+secure_filename(f.filename), fruits_classes=fruits_classes, preds=preds)



if __name__ == "__main__":
    app.run(port="5001",debug=True)
    