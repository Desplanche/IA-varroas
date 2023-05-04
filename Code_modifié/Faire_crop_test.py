# chargement de l'image
image = cv2.imread(fname)

if image is None: 
    result = "Image is empty!!"
else: 
    result = "Image is not empty!!"
    print(' Image ',filename)
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    angle = 90
    scale = 1
    
    # dimensions du crop final de 22x22
    crop_size = 22
    
    # demi largeur du crop final
    h1 = crop_size // 2
    
    # boucle pour les différents angles de rotation
    for angle in range(0, 360, 90):
        # matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # rotation de l'image
        rotated = cv2.warpAffine(image, M, (w, h))
        
        # découpe de l'image
        for i in range(h1, h-h1, crop_size):
            for j in range(h1, w-h1, crop_size):
                crop_img = rotated[i-h1:i+h1, j-h1:j+h1]
                fname1 = folder1 + "crop_" + str(num_crop) + ".jpg"
                print(fname1)
                cv2.imwrite(fname1, crop_img)
                num_crop += 1
