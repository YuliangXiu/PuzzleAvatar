import cv2
import os

image_folder = '/home/yxiu/Code/PuzzleAvatar/results/full/human/yuliang_short_shirt/texture/validation'
video_name = '/home/yxiu/Code/PuzzleAvatar/results/full/human/yuliang_short_shirt/texture/validation/head2-rgb.avi'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith("tex.png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))[800:,800:1600]
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter(video_name, fourcc, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image))[800:,800:1600])

cv2.destroyAllWindows()
video.release()