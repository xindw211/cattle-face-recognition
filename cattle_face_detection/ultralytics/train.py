from ultralytics import YOLO

model = YOLO('./cfg/models/v8/my_yolov8.yaml')

model.info()
results1 = model.train(data='./cfg/datasets/cow.yaml')
results2 = model.val()
results3 = model.val(split='test')
success = model.export(format='onnx')
