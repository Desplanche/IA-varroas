import sys
import os
import cv2

num_crop = 0
folder0 = "./varroas/"
folder1 = "./varoas_rotation/"

image_names = os.listdir(folder0)

for filename in image_names:
    fname = folder0 + filename
    print(fname)
    image = cv2.imread(fname)  # original image
    if image is None:
        result = "Image is empty!!"
    else:
        result = "Image is not empty!!"
        print(' Image ', filename)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        angle = 90
        scale = 1
        h1 = 20  # half width of the final crop => crop of 22
        b1 = int(w / 2 - h1)  # left corner
        b2 = int(w / 2 + h1)  # right corner
        a1 = int(h / 2 - h1)  # top corner
        a2 = int(h / 2 + h1)  # bottom corner

        for angle in range(0, 360, 90):
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))
            crop_img = rotated[a1:a2, b1:b2]

            fname1 = folder1 + "crop_" + str(num_crop) + ".jpg"
            print(fname1)
            cv2.imwrite(fname1, crop_img)

            # Display the image
            #cv2.imshow("Crop Image", crop_img)
            #cv2.waitKey(0)  # Wait for a key press to close the image window
            #cv2.destroyAllWindows()

            num_crop += 1
