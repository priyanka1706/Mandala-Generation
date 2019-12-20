import numpy as np
import cv2
import os.path

w_sec=198

path = "created2/auto"
if os.path.isdir(path):
    print("exists")
else:
    os.mkdir(path)

def create_mandala_half(image_tot, x1, x2, y1, y2):
    image = image_tot[x1:x2, y1:y2] #[400:, 600:]
    image2 = cv2.flip(image, 1)
    image = np.append(image, image2, axis=1)
    image2 = cv2.flip(image, 0)
    image = np.append(image, image2, axis=0)
    
    #circular mask to make final mandala circular
    sh=image.shape
    mask = np.zeros(sh, dtype=np.uint8)
    cv2.circle(mask, (int(sh[0]/2),int(sh[1]/2)), int(sh[1]/2), (255,255,255), -1)
    image[np.where(mask==0)]=255
    
    name = path+"\\"+"image-quart-"+str(sets)+"-"+str(no)+".jpg"
    cv2.imwrite(name, image)   
    
def create_mandala(image_tot, x1, x2, y1, y2):
    img = image_tot[x1:x2, y1:y2] #[400:, 600:]
    
    #create mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    ignore_mask_color = (255,)*1
    corners = [[0,0],[0,w_sec],[w_sec,0]]
    corners = np.array([corners])
    cv2.fillPoly(mask, corners, ignore_mask_color)
    
    #area we want
    img_out = cv2.bitwise_and(img,img,mask=mask)

    #flipped image roi
    M = cv2.getRotationMatrix2D((w_sec/2,w_sec/2), 270, 1.0)
    img_flip_out = cv2.warpAffine(img, M, (w_sec, w_sec))
    
    #fixing triangle cross sections together
    for i in range(w_sec - 1):
        for j in range (int(w_sec * float(w_sec - i) / w_sec)):#(w_sec-int(w_sec * float(w_sec - i) / w_sec), 0, -1):
            img_out[w_sec-1-i, w_sec-1-j] = img_flip_out[i, w_sec-1-j]
            #img_out[i,j] = img_flip_out[i,j]
            
    #stitching images together
    img2 = cv2.flip(img_out, 0)
    img_fin = np.append(img_out, img2, axis=0)
    
    img2 = cv2.flip(img_fin, -1)
    img_fin = np.append(img2, img_fin, axis=1)
    
    #circular mask to make final mandala circular
    sh=img_fin.shape
    mask = np.zeros(sh, dtype=np.uint8)
    cv2.circle(mask, (int(sh[0]/2),int(sh[1]/2)), int(sh[1]/2), (255,255,255), -1)
    img_fin[np.where(mask==0)]=255
    
    name = path+"\\"+"image-"+str(sets)+"-"+str(no)+".jpg"
    cv2.imwrite(name, img_fin)        
    
#for loop get seed if reproducible data set is needed
#seed = 0

#to do: create image
#try to randomize point1, point2, r ??

for sets in range(0,10):
    image_tot = np.zeros((600, 800), np.uint8)
    image_tot[:,:] = 255

    for i in range(0,18):
        p1 = (np.random.randint(800),np.random.randint(600))
        p2 = (np.random.randint(800),np.random.randint(600))
        cv2.line(image_tot,p1,p2,(0,0,0),2)
    for i in range(0,7):    
        p1 = (np.random.randint(800),np.random.randint(600))
        p2 = (np.random.randint(800),np.random.randint(600))
        cv2.rectangle(image_tot,p1,p2,(0,0,0),3)
    for i in range(0,60):    
        p1 = (np.random.randint(800),np.random.randint(600))
        r = np.random.randint(100)
        cv2.circle(image_tot,p1, r, (0,0,0), 2)

    #cv2.imwrite("main"+str(sets)+".jpg", image_tot)
    #cv2.imshow("fin", image_tot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    no=1
    for i in range(0,800,200):
        for j in range(0,600,200):
            create_mandala(image_tot, j, j+199, i, i+199)
            create_mandala_half(image_tot, j, j+199, i, i+199)
            no+=1
                
    #flip image and redo 
    image_tot=cv2.flip(image_tot,0)       
    for i in range(0,800,200):
        for j in range(0,600,200):
            create_mandala(image_tot, j, j+199, i, i+199)
            create_mandala_half(image_tot, j, j+199, i, i+199)
            no+=1
    
    #flip image and redo 
    image_tot=cv2.flip(image_tot,1)       
    for i in range(0,800,200):
        for j in range(0,600,200):
            create_mandala(image_tot, j, j+199, i, i+199)
            create_mandala_half(image_tot, j, j+199, i, i+199)
            no+=1