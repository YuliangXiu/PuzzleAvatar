import numpy as np
import cv2


src_img = cv2.imread("data/human/yuliang/image/03.jpg")
mask_1 = cv2.imread("data/human/yuliang/mask/03_pants.png", cv2.IMREAD_GRAYSCALE)
mask_2 = cv2.imread("data/human/yuliang/mask/03_shoes.png", cv2.IMREAD_GRAYSCALE)

final_img = src_img * (mask_1[..., None] / 255) + src_img * (mask_2[..., None] / 255)
final_img += ((mask_1 + mask_2) == 0)[..., None] * 0.5 * 255.0


cv2.imwrite("tmp/overlap.jpg", final_img)