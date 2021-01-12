from tkinter import *
import pyscreenshot as ImageGrab
from numpy import asarray
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image, ImageOps



canvas_width = 560
canvas_height = 560

def draw(event):
    color='white'
    if event.x > 0 and event.x < canvas_width and\
       event.y > 0 and event.y < canvas_height:
        x0,y0=(event.x-6), (event.y-6)
        x1,y1=(event.x+6), (event.y+6)
        my_canvas.create_oval(x0,y0,x1,y1,fill=color,outline=color)


def get_canvas():
    x2=master.winfo_rootx()+my_canvas.winfo_x()
    y2=master.winfo_rooty()+my_canvas.winfo_y()
    x1=x2+my_canvas.winfo_width()
    y1=y2+my_canvas.winfo_height()
    ImageGrab.grab().crop((x2+50,y2+50,x1-50,y1-50)).save("./drawn_number.jpg")
    my_canvas.delete('all')
    my_canvas.create_rectangle(40, 40, canvas_width-40, canvas_height-40, fill="black")
    image_to_numpy()


# Need it to be (28, 28) with 2 dimensions

def image_to_numpy():
    image = Image.open('drawn_number.jpg')
    image = image.convert('L')
    image = np.array(image.resize((28,28)))
    image = tf.keras.utils.normalize(image, axis=1) 
    image = np.array(image)
    #Image.fromarray(image).save('drawn_number.jpg')
    image_test(image)


def image_test(image):
    model = tf.keras.models.load_model('saved_model/my_model')
    image = np.reshape(image, (1,784))
    action = model.predict(image)
    max_value = np.argmax(action)
    print(f"Was that a {max_value}?")

master=Tk()
master.title('Draw Number (0-9)')
master.resizable(False, False)

my_canvas=Canvas(master, width=canvas_width, height=canvas_height, bg="seashell3")
my_canvas.pack(expand=YES, fill=BOTH)
my_canvas.bind('<B1-Motion>',draw)

my_canvas.create_rectangle(40, 40, canvas_width-40, canvas_height-40, fill="black")

# Button
guess_button = Button(text='Guess', command=get_canvas)
guess_button.place(x=canvas_width-45,y=canvas_height-30)

master.mainloop()



