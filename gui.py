import os
import time
import time
from ctypes import windll
from enum import Enum
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter.font import Font

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
# from natsort import natsorted
from pyscreenshot import grab

Path('saved_images').mkdir(exist_ok=True)  # 스크린샷 저장할 경로 생성
CAPTURED_IMAGES_PATH = os.getcwd() / Path("saved_images")
BBOX_RESULT_PATH = os.getcwd() / Path("yolov5/runs/detect")
SEG_RESULT_PATH = os.getcwd() / Path("Pytorch-UNet/output")

def snapshot():
    # Warning:
    # 같은 이름으로 저장된다면 실제디렉토리엔 하나만 저장되지만, capture_lb엔 계속 추가된다.
    filename = time.strftime("%Y-%m-%d-%H-%M-%S") + ".png"
    dir = "saved_images/" + filename

    target_img = ImageTk.getimage(video10.img)  # capture segmentation field
    target_img.save(dir)
    print("Screenshot Saved..", type(target_img))
    capture_lb.insert(END, filename)


# 캡쳐 사진으로 교체
def popup_saved_image(filename):
    print(filename)
    FILE_DIR = CAPTURED_IMAGES_PATH / Path(filename)
    img = Image.open(FILE_DIR)
    img = ImageTk.PhotoImage(image=img)
    video11.img = img
    video11.configure(image=img)


def find_directory():
    img_source = filedialog.askopenfilename(initialdir="./test_data", title="Select A File")
    if img_source != "":
        img = Image.open(img_source)
        width = int(img.size[0])
        height = int(img.size[1])
        img = img.resize((width, height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image=img)
        video00.img = img
        video00.configure(image=img)
        return img_source
    return None


def load_resize_result(crack_result, video_loc):
    width = int(crack_result.size[0] )
    height = int(crack_result.size[1] )
    crack_result = crack_result.resize((width, height), Image.ANTIALIAS)
    crack_result = ImageTk.PhotoImage(image=crack_result)
    video_loc.img = crack_result
    video_loc.configure(image=crack_result)

def predict_bbox_yolov5(img_source):
    print("==bbox predict....==")
    # 추론한 결과가 BBOX_RESULT_PATH 경로에 저장됨
    os.system(
        'python ./yolov5/detect.py --weight ./yolov5/runs/train/results/weights/best.pt --img 416 --conf 0.3 --source ' + img_source)

    # TODO : natsort를 사용할 수 있도록 python<3.6 으로 downgrade 필요
    # print(natsorted(os.listdir(BBOX_RESULT_PATH)))
    last_folder = os.listdir(BBOX_RESULT_PATH)[-1]
    last_img = os.listdir(BBOX_RESULT_PATH / Path(last_folder))[-1]
    last_result = BBOX_RESULT_PATH / Path(last_folder) / Path(last_img)
    print("last_result", last_result)
    crack_result = Image.open(last_result)
    load_resize_result(crack_result, video01)


def predict_seg_unet(img_source):
    print("==seg predict....==")
    seg_result_name = img_source.split("/")[-1].split(".")[0]+".png"
    os.system('python ./Pytorch-UNet/predict.py -m ./Pytorch-UNet/MODEL.pth -i '+img_source+'  -o '+os.path.join(SEG_RESULT_PATH,seg_result_name))
    print(seg_result_name, os.path.join(SEG_RESULT_PATH,seg_result_name))
    crack_result = Image.open(os.path.join(SEG_RESULT_PATH,seg_result_name))
    load_resize_result(crack_result, video10)

def predict_all():
    img_source = find_directory()
    if img_source !=None:
        start1 = time.time()
        predict_bbox_yolov5(img_source)
        end1 = time.time()
        print(f"bbox: {end1 - start1:.5f} sec")

        start2 = time.time()
        predict_seg_unet(img_source)
        end2 = time.time()
        print(f"seg: {end2 - start2:.5f} sec")

        print("total: %.5f"%(end1-start1 + end2 -start2))

# -------------tkinter-----------------
window = Tk()

# 창을 screen 중간에 열기
w = 1200  # width for the Tk root
h = 650  # height for the Tk root

ws = window.winfo_screenwidth()  # width of the screen
hs = window.winfo_screenheight()  # height of the screen

x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)

window.geometry('%dx%d+%d+%d' % (w, h, x, y))
window.minsize(w, h)
window.configure(bg="white")
window.title("캡스톤디자인 NETFLEX팀 경계검출")
window.bind('<Escape>', lambda e: window.quit())  # esc로 창 종료
window.resizable(True, True)

# Canvas 3x2
canvas_frame = Frame(window, width=w, height=h)
canvas_frame.pack(fill=BOTH, expand=YES)
canvas_frame.columnconfigure(0, weight=3)
canvas_frame.columnconfigure(1, weight=3)
canvas_frame.rowconfigure(0, weight=3)
canvas_frame.rowconfigure(1, weight=3)
canvas_frame.rowconfigure(2, weight=1)

# Top layout
video00 = Label(canvas_frame, bg="lightslategray", borderwidth=2, relief="ridge")
video01 = Label(canvas_frame, bg="lightslategray", borderwidth=2, relief="ridge")
video10 = Label(canvas_frame, bg="lightslategray", borderwidth=2, relief="ridge")
video11 = Label(canvas_frame, bg="lightslategray", borderwidth=2, relief="ridge")

video00.grid(row=0, column=0, sticky=NSEW)
video01.grid(row=0, column=1, sticky=NSEW)
video10.grid(row=1, column=0, sticky=NSEW)
video11.grid(row=1, column=1, sticky=NSEW)

# bottom layout
scaleFont = Font(family='Tahoma', size=10, weight='bold')

bottom_layout = Frame(canvas_frame, bg="white")
bottom_layout.grid(row=2, columnspan=2, sticky=NSEW)
bottom_layout.columnconfigure(0, weight=6)
bottom_layout.columnconfigure(1, weight=1)
bottom_layout.rowconfigure(0, weight=1)

# Parameter layout 2x2
param_layout = Frame(bottom_layout)
param_layout.grid(row=0, column=0, sticky=NSEW)
param_layout.rowconfigure(0, weight=1)
param_layout.rowconfigure(1, weight=3)
param_layout.columnconfigure(0, weight=1)
param_layout.columnconfigure(1, weight=7)

# row=0
blur_frame = Frame(param_layout, bg="white", borderwidth=1, relief=SUNKEN)
blur_frame.grid(row=0, column=0, sticky=NSEW)
filter_frame = Frame(param_layout, bg="white", borderwidth=1, relief=SUNKEN)
filter_frame.grid(row=0, column=1, sticky=NSEW)

upload_img_btn = Button(filter_frame, text="upload image",
                        bg="midnightblue",
                        fg="white",
                        font=scaleFont,
                        command=lambda: predict_all())  # find_directory())
upload_img_btn.pack(side="right")

# Capture layout
capture_layout = Frame(bottom_layout, bg="white")
capture_layout.grid(row=0, column=1, rowspan=2, sticky="NSEW")

capture_btn = Button(capture_layout,
                     text="Save Image",
                     font=scaleFont,
                     fg="white",
                     bg="#4535AA",
                     activeforeground="#009888",
                     borderwidth=3,
                     command=lambda: snapshot(),
                     relief="groove")
capture_btn.pack(pady=3)

scrollbar = Scrollbar(capture_layout)
scrollbar.pack(side="right", fill="y")

capture_lb = Listbox(capture_layout, yscrollcommand=scrollbar.set)
capture_lb['bg'] = "black"
capture_lb['fg'] = "lime"
capture_lb['font'] = scaleFont
capture_lb.pack(pady=3, fill="x", expand="yes")
capture_lb.bind("<<ListboxSelect>>", lambda x: popup_saved_image(capture_lb.get(capture_lb.curselection())))

for file in os.listdir(CAPTURED_IMAGES_PATH):
    capture_lb.insert(END, file)

scrollbar["command"] = capture_lb.yview
# -----------------------------------

if __name__ == "__main__":
    print("start")
    # cam_thread()
    window.mainloop()
    print("end")
