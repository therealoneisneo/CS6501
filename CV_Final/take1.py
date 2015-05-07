import Tkinter
from Tkinter import *
import numpy as np
import Image, ImageTk
import SphereTop as st

class App:
    
    panel_w = 0.0
    panel_h = 0.0
        
    def __init__(self, root, w, h):
        self.panel_w = w
        self.panel_h = h
        self.root = root
        self.mouse_pressed = False
        f = Tkinter.Frame(width = self.panel_w, height = self.panel_h, background="bisque")
        f.pack(padx=100, pady=100)
        f.bind("<ButtonPress-1>", self.OnMouseDown)
        f.bind("<ButtonRelease-1>", self.OnMouseUp)

    def do_work(self):
        x = self.root.winfo_pointerx()
        y = self.root.winfo_pointery()
        xx = self.root.winfo_rootx()
        yy = self.root.winfo_rooty()
        x = x-xx-100
        y = y-yy-100
        
        x -= self.panel_w / 2
        y -= self.panel_h / 2
        
        
        
        rescale(paneldata, xmax, ymax)
        
        (index1, index2, index3) = st.Interpolate(paneldata, x, y)
        
        
        
        
        s1 = "%s.jpg" %(index1[0])
        s2 = "%s.jpg" %(index2[0])
        s3 = "%s.jpg" %(index3[0])
        w1 = index1[1]
        w2 = index2[1]
        w3 = index3[1]
        
        print s1
        img1=Image.open(s1)
        img2=Image.open(s2)
        img3=Image.open(s3)
        
        weight = w2 / (w2 + w1)

        imgInter = Image.blend(img1, img2, weight)
        weight = w3 / (w1 + w2 + w3)
        img = Image.blend(imgInter, img3, weight)
        imgTk = ImageTk.PhotoImage(img)
        l.configure(image = imgTk)
        l.image = imgTk
        print "button is being pressed... (%s,%s)" % (x, y)

    def OnMouseDown(self, event):
        self.mouse_pressed = True
        self.poll()

    def OnMouseUp(self, event):
        self.root.after_cancel(self.after_id)

    def poll(self):
        if self.mouse_pressed:
            self.do_work()
            self.after_id = self.root.after(250, self.poll)
            
    def rescale(self, data, xmax, ymax):
        n = len(data)
        factor_x = self.panel_w / 2
        factor_y = self.panel_h / 2
        for i in range(n):
            data[i, 0] = data[i, 0] * factor_x / xmax
            data[i, 1] = data[i, 1] * factor_y / ymax
            
root = Tk()


w = 200
h = 200

app = App(root, w, h)


img=Image.open('0.jpg')
imgTk=ImageTk.PhotoImage(img)

l=Label(root,image=imgTk)

l.pack()
root.mainloop()
