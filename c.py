import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = cv2.imread('picture.jpg')
results = model(img)
results.save()

result = results.pandas().xyxy[0].to_numpy()
result = [item for item in result if item[6]=='person']

tmp_img = cv2.imread('picture.jpg')
print(tmp_img.shape)

for i in range(len(result)):
    cropped = tmp_img[int(result[i][1]):int(result[i][3]), int(result[i][0]):int(result[i][2])]
    print(cropped.shape)
    cv2.imwrite(f'people{i}.png', cropped)
    
    
    
    
#이미지에서 사람들을 잘라 people1.png, people2.png… 로 저장하세요    
# import torch
# import cv2
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# img = cv2.imread('people.jpg')

# results = model(img)
# result = results.pandas().xyxy[0].to_numpy()
# result = [ item for item in result if item[6] == 'person']

# tmp_img = cv2.imread('people.jpeg')

# for i, item in enumerate(result):
#     cropped = tmp_img[int(item[1]):int(item[3]), int(item[0]):int(item[2])]
#     cv2.imwrite(f'people{i}.png', cropped)
