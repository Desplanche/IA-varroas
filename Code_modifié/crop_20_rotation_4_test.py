import sys,os,json
import cv2
num_crop = 0
folder0= "base/varroas/"
folder1= "base/varoas_rotation/"
# E:\creation_model_5\neo_non_varroas
image_names=os.listdir(folder0)

def crop_image(img):
    # Ouvrir l'image

    # Récupérer les dimensions de l'image
    width, height = image.size

    # Calculer les coordonnées de recadrage
    left = (width - 11) // 2
    top = (height - 11) // 2
    right = left + 11
    bottom = top + 11

    # Recadrer l'image
    cropped_image = crop_image.crop((left, top, right, bottom))

    # Retourner l'image recadrée
    return cropped_image


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
            cropped_image = crop_image(crop_img)

            # N_rand = random.randint(1000,9999)
            fname1 = folder1+"crop_"+str(num_crop)+".jpg"    
            print (fname1)
            cv2.imwrite(fname1,cropped_image)
            num_crop = num_crop +1
