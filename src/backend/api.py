from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from torchvision.utils import save_image
import uvicorn
import torch
from utils import MainModel


app = FastAPI()

@app.on_event("startup")
def load_model():
    # import mlflow
    # global loaded_model
    #
    # mlflow.set_tracking_uri('https://dagshub.com/RafaJBZ/Chromatic-charm.mlflow')
    # logged_model = 'runs:/76d63b43be4c45cbae8aa46d79291091/GANv2'
    # loaded_model = mlflow.pyfunc.load_pyfunc(logged_model)
    global loaded_model
    loaded_model = MainModel()
    loaded_model.load_weights('./gan-weights.pth')

@app.post('/api/v1/colorize')
async def colorize(file: UploadFile = File(...)):
    size_transform = transforms.Resize(size=(256,256),
                                        interpolation=transforms.InterpolationMode.NEAREST)

    decoded_file = await file.read()
    decoded_tensor = torch.frombuffer(decoded_file, dtype=torch.uint8)
    img = decode_image(decoded_tensor, mode=ImageReadMode.GRAY)
    img = size_transform(img)
    img = img / 255

    tensor = img.resize_(1,1,256,256)
    pred = loaded_model.test_predict(tensor)

    img_file = 'img.jpg'
    save_image(pred[0], fp=img_file)
    return FileResponse(img_file)

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=5050, log_level='info')
