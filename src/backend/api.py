from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import uvicorn

app = FastAPI()

@app.on_event("startup")
def load_model():
    import mlflow
    global logged_model

    mlflow.set_tracking_uri('')
    logged_model = ''
    logged_model = mlflow.pyfunc.load_model(logged_model)


@app.post('/api/v1/colorize')
async def colorize(file: UploadFile = File(...)):
    size_transform = transforms(size=(256,256),
                                interpolation=transforms.InterpolationMode.NEAREST)

    img = await read_image(file, mode=ImageReadMode.RGB)
    img = size_transform(img)

    tensor = img.resize_(1,256,256)
    logged_model.predict(tensor)

    return None

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=5050, log_level='info')
