import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://teamsparta.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F396579ac-a81f-4073-9409-248ba51bc5ea%2FUntitled.jpeg?table=block&id=e70a6c23-18cd-4bc6-8c46-b585967dfd56&spaceId=83c75a39-3aba-4ba4-a792-7aefe4b07895&width=2000&userId=&cache=v2']  # batch of images

results = model(imgs)

print(results.xyxy[0], results.xyxy[0][0][0].item())  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)


tmp_img = cv2.imread('picture.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][0][0].item()), int(results.xyxy[0][0][1].item())), (int(results.xyxy[0][0][2].item()), int(results.xyxy[0][0][3].item())), (255,255,255))
cv2.imwrite('result1.png', tmp_img)

tmp_img = cv2.imread('picture.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][1][0].item()), int(results.xyxy[0][1][1].item())), (int(results.xyxy[0][1][2].item()), int(results.xyxy[0][1][3].item())), (255,255,255))
cv2.imwrite('result2.png', tmp_img)

tmp_img = cv2.imread('picture.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][2][0].item()), int(results.xyxy[0][2][1].item())), (int(results.xyxy[0][2][2].item()), int(results.xyxy[0][2][3].item())), (255,255,255))
cv2.imwrite('result3.png', tmp_img)

tmp_img = cv2.imread('picture.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][3][0].item()), int(results.xyxy[0][3][1].item())), (int(results.xyxy[0][3][2].item()), int(results.xyxy[0][3][3].item())), (255,255,255))
cv2.imwrite('result4.png', tmp_img)

tmp_img = cv2.imread('picture.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][4][0].item()), int(results.xyxy[0][4][1].item())), (int(results.xyxy[0][4][2].item()), int(results.xyxy[0][4][3].item())), (255,255,255))
cv2.imwrite('result5.png', tmp_img)



#사람을 찾아 하얀색 네모를 그려 result.png 로 저장
# import torch
# import cv2
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# img = cv2.imread('people.jpg')

# results = model(img)
# result = results.pandas().xyxy[0].to_numpy()
# result = [ item for item in result if item[6] == 'person']

# tmp_img = cv2.imread('people.jpeg')

# for i, item in enumerate(result):
#     cv2.rectangle(tmp_img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255,255,255))
# cv2.imwrite('result.png', tmp_img)