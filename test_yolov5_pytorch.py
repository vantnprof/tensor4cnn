import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)