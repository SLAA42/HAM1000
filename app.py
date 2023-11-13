!import fastai
import fastai
from fastai.vision.all import*
import gradio as gr

def is_cat(x):returnx[0].isupper()

learn=load_learner('my_export_HAM10000.pkl')

categories=('akiec','bcc','bkl','df','mel','nv','vasc')

def classify_img(img):
    pred,idx,probs=learn.predict(img)
    return dict(zip(categories,map(float,probs)))
    
image =gr.inputs.Image(shape=(192,192))
label =gr.outputs.Label()
example=['mel.jpg','akiec.jpg','vasc.jpg']
intf=gr.Interface(fn=classify_img,inputs=image,outputs=label,examples=example)
intf.launch(inline=False)