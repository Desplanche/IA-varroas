import sys,os,json
import cv2
num_crop = 0
folder0= "./filtre_a_crop/"
folder1= "./rotation_resize/"
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

            height, width, _ = crop_img.shape

            # Découpage en carrés de 40x40 pixels
            step = 40
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Récupération du carré
                    square = crop_img[y:y+step, x:x+step]
        
            # Affichage du carré
            cv2.imshow("Square", square)

            # N_rand = random.randint(1000,9999)
            fname1 = folder1+"crop_"+str(num_crop)+".jpg"    
            print (fname1)
            cv2.imwrite(fname1,crop_img)
            num_crop = num_crop +1
