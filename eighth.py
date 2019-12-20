import numpy as np
import cv2
import os.path
from tkinter import *
from PIL import Image, ImageGrab

path = "created2"
if os.path.isdir(path):
    print("exists")
else:
    os.mkdir(path) 
    
w=300 #multiple of 200
h=300
w_sec=199

def create_mandala(img):
    #img = image_tot[x1:x2, y1:y2] #[400:, 600:]
    
    #create mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    channel_count = img.shape[2]  
    ignore_mask_color = (255,)*channel_count
    corners = [[0,0],[0,w_sec],[w_sec,0]]
    corners = np.array([corners])
    cv2.fillPoly(mask, corners, ignore_mask_color)
    
    #area we want
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_out = cv2.bitwise_and(img,img,mask=mask)
    img_out = img_out[:w_sec, :w_sec]

    #flipped image roi
    M = cv2.getRotationMatrix2D((w_sec/2,w_sec/2), 270, 1.0)
    img_flip_out = cv2.warpAffine(img, M, (w_sec, w_sec))
    
    #fixing triangle cross sections together
    for i in range(w_sec - 1):
        for j in range (int(w_sec * float(w_sec - i) / w_sec)):#(w_sec-int(w_sec * float(w_sec - i) / w_sec), 0, -1):
            img_out[w_sec-1-i, w_sec-1-j] = img_flip_out[i, w_sec-1-j]
            #img_out[i,j] = img_flip_out[i,j]
    
    #cv2.imwrite("flip_diag.jpg", img_out)
    
    #stitching images together
    img2 = cv2.flip(img_out, 0)
    img_fin = np.append(img_out, img2, axis=0)
    #cv2.imwrite("flip1.jpg", img_fin)
    
    img2 = cv2.flip(img_fin, -1)
    img_fin = np.append(img2, img_fin, axis=1)
    #cv2.imwrite("flip2.jpg", img_fin)
    
    #circular mask to make final mandala circular
    sh=img_fin.shape
    mask = np.zeros(sh, dtype=np.uint8)
    cv2.circle(mask, (int(sh[0]/2),int(sh[1]/2)), int(sh[1]/2), (255,255,255), -1)
    img_fin[np.where(mask==0)]=255
    #cv2.imwrite("final.jpg", img_fin)
      
    return img_fin

class Scribble(object):

    def __init__(self):
        self.root = Tk()

        Label(self.root, text = 'Draw your custom Mandala!', font =('Verdana', 15)).grid(column=1, columnspan=5, row=0)
        
        self.c = Canvas(self.root, bg='white', width=w, height=h)
        self.c.grid(row=2, columnspan=5)
        
        self.size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.size_button.grid(row=1, column=2)

        self.c.create_line(0,h,w,0, fill="black")
        
        self.prevx = None
        self.prevy = None
        self.c.bind('<B1-Motion>', self.drawing)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind('<Leave>', self.save_image)

        self.root.mainloop()

    def drawing(self, event):
        self.line_width = self.size_button.get()
        if self.prevx and self.prevy:
            self.c.create_line(self.prevx, self.prevy, event.x, event.y,
                               width=self.line_width,
                               capstyle=ROUND, smooth=TRUE)
        self.prevx = event.x
        self.prevy = event.y

    def reset(self, event):
        self.prevx, self.prevy = None, None
        
    def save_image(self, event):
        print("entered save")
        self.getter(self.c)
        
    def getter(self,widget):
        '''
        source: https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image
        
        root.winfo_rootx() and the root.winfo_rooty() get the pixel position of the top left of the overall root window.
        widget.winfo_x() and widget.winfo_y() are added to, basically just get the pixel coordinate of the top left hand 
        pixel of the widget which you want to capture (at pixels (x,y) of your screen).

        find the (x1,y1) which is the bottom left pixel of the widget
        ImageGrab.grab() makes a printscreen, and then crop it to only get the bit containing the widget
        '''
        x=self.root.winfo_rootx()+widget.winfo_x()
        y=self.root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save(path+"/Main.jpg") 

if __name__ == '__main__':
    Scribble()
    
    image_tot=cv2.imread(path+"/Main.jpg")   
    #image_tot = cv2.imread("subimage.jpg")
    image = create_mandala(image_tot)
    name = path+"/eighth.jpg"
    cv2.imwrite(name, image)
    cv2.imshow("fin", image)
    cv2.waitKey()
    cv2.destroyAllWindows()