import sys,os,json
import cv2
num_crop = 0
folder0= "varroas/"
folder1= "varoas_rotation/"
# E:\creation_model_5\neo_non_varroas
image_names=os.listdir(folder0)

for filename in image_names:
    fname = folder0+filename
    print (fname)
    image = cv2.imread(fname)  # image d'origine   
    if image is None: 
       result = "Image is empty!!"
    else: 
        result = "Image is not empty!!"
        print(' Image ',filename)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        angle = 90
        scale = 1
        h1 = 20 # demi largeur du crop final => crop de 22
        b1 = int(w/2-h1) # coin à gauche 
        b2 = int(w/2+h1) # coin à gauche 
        a1 = int(h/2-h1) # coin à gauche 
        a2 = int(h/2+h1) # coin à gauche 

        for angle in range(0,360,90):
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))
            crop_img = rotated[a1:a2,b1:b2]

            # N_rand = random.randint(1000,9999)
            fname1 = folder1+"crop_"+str(num_crop)+".jpg"    
            print (fname1)
            cv2.imwrite(fname1,crop_img)
            num_crop = num_crop +1
