from gettext import find
import math
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

classes = [39, 61, 26]
imageClassList = {'0': 'Jet', '1': 'Drone', '2': 'Helicopter'}  #Сюда указать классы
sess = onnxruntime.InferenceSession(r'/Users/kravtandr/Desktop/DeepLearning/my/dz1/media/models/cifar100_DZ_82.onnx') #<-Здесь требуется указать свой путь к модели

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction, out = predictImageData(modelName, '.'+filePathName)
    hist = draw_histogram(out)
    info = imgInfo("/Users/kravtandr/Desktop/DeepLearning/my/dz1/"+filePathName)
    context = {'scorePrediction': scorePrediction, 'image': filePathName, 'hist': hist, 'info_size': info['Image Size'], 'info_format':  info['Image Format'], 'info_weight': info['Size']}
    return render(request, 'scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((64, 64), Image.ANTIALIAS))
    sess = onnxruntime.InferenceSession(r'/Users/kravtandr/Desktop/DeepLearning/my/dz1/media/models/cifar100_DZ_82.onnx') #<-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    classes = sess.run(None, {'input': np.asarray([img]).astype(np.float32)})
    score = imageClassList[str(outputOFModel)]
    if score == "Jet":
        score = "Истребитель"
    elif score == "Drone":
        score = "Дрон"
    elif score == "Helicopter":
        score = "Вертолет"
    return score, classes




def draw_histogram(predictions):
    def softmax(x):
        return(np.exp(x)/np.exp(x).sum())
    
    data = softmax(predictions[0])
    
# 
    df = pd.DataFrame(data[0])
    import matplotlib
    matplotlib.use('agg')
    fig, ax = plt.subplots()

    classes = ['Истребитель', 'Дрон', 'Вертолет']
    counts = data[0]
    bar_labels = ['Истребитель', 'Дрон', 'Вертолет']
    bar_colors = ['tab:red', 'tab:blue', 'tab:green']

    ax.bar(classes, counts, label=bar_labels, color=bar_colors)

    ax.legend(title='Classes')

    # plt.show()
    plt.savefig('media/images/foo.png')


    fs = FileSystemStorage()
    img = "/media/images/foo.png"
    


    return img

def imgInfo(imagename):
    from PIL import Image
    from PIL.ExifTags import TAGS

    image = Image.open(imagename)

    info_dict = {
        "Filename": image.filename,
        "Image Size": image.size,
        "Image Height": image.height,
        "Image Width": image.width,
        "Image Format": image.format,
        "Image Mode": image.mode,
        "Image is Animated": getattr(image, "is_animated", False),
        "Frames in Image": getattr(image, "n_frames", 1),
        'Size': os.stat(imagename).st_size
    }

    for label,value in info_dict.items():
        print(f"{label:25}: {value}")

    return info_dict
   