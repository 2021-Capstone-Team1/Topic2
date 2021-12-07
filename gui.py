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
from pyscreenshot import grab
from tkinter.ttk import Progressbar

# Path('saved_images').mkdir(exist_ok=True)  # 스크린샷 저장할 경로 생성
BBOX_RESULT_PATH = os.getcwd() / Path("yolov5_object_size/runs/detect/bbox")
SEG_RESULT_PATH = os.getcwd() / Path("Pytorch-UNet/output")
SIZE_RESULT_PATH = os.getcwd() / Path("yolov5_object_size/runs/detect/size")


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
    width = int(crack_result.size[0])
    height = int(crack_result.size[1])
    crack_result = crack_result.resize((width, height), Image.ANTIALIAS)
    crack_result = ImageTk.PhotoImage(image=crack_result)
    video_loc.img = crack_result
    video_loc.configure(image=crack_result)

def progress(pb, value):
    pb['value'] = value

def predict_bbox_yolov5(img_source):
    pb01.pack(expand=True)

    progress(pb01, 70)
    pb01.update()
    print("==bbox predict....==")

    # 추론한 결과가 BBOX_RESULT_PATH 경로에 저장됨
    os.system(
        'python ./yolov5_object_size/detect.py --weight ./yolov5_object_size/runs/train/results/weights/best.pt --img 416 --conf 0.3 --source ' + img_source)
    # progress(pb01, 80)
    # pb01.update()

    last_result = BBOX_RESULT_PATH / Path(img_source.split("/")[-1])
    print(last_result)
    # progress(pb01, 90)
    # pb01.update()

    crack_result = Image.open(last_result)

    load_resize_result(crack_result, video01)
    # progress(pb01, 100)
    # pb01.update()
    pb01.pack_forget()


def predict_seg_unet(img_source):
    pb10.pack(expand=True)
    progress(pb10, 70)
    pb10.update()
    print("==seg predict....==")

    seg_result_name = img_source.split("/")[-1].split(".")[0] + ".png"
    os.system('python ./Pytorch-UNet/predict.py -m ./Pytorch-UNet/MODEL.pth -i ' + img_source + '  -o ' + os.path.join(
        SEG_RESULT_PATH, seg_result_name))
    # progress(pb10, 80)
    # pb10.update()

    crack_result = Image.open(os.path.join(SEG_RESULT_PATH, seg_result_name))
    # progress(pb10, 90)
    # pb10.update()

    load_resize_result(crack_result, video10)
    # progress(pb10, 100)
    # pb10.update()
    pb10.pack_forget()


def prdict_size_yolov5(img_source):
    pb11.pack(expand=True)
    print("==size predict....==")
    progress(pb11, 70)
    pb11.update()

    last_result = SIZE_RESULT_PATH / Path(img_source.split("/")[-1])
    # progress(pb11, 80)
    # pb11.update()

    crack_result = Image.open(last_result)
    # progress(pb11, 90)
    # pb11.update()

    load_resize_result(crack_result, video11)
    # progress(pb11, 100)
    # pb11.update()
    pb11.pack_forget()

def predict_all():
    img_source = find_directory()
    if img_source is not None:
        print(img_source)
        start1 = time.time()
        predict_bbox_yolov5(img_source)
        end1 = time.time()
        print(f"bbox: {end1 - start1:.5f} sec")

        start2 = time.time()
        predict_seg_unet(img_source)
        end2 = time.time()
        print(f"seg: {end2 - start2:.5f} sec")

        start3 = time.time()
        prdict_size_yolov5(img_source)
        end3 = time.time()
        print(f"size: {end2 - start2:.5f} sec")
        print("total: %.5f" % (end1 - start1 + end2 - start2 + end3 - start3))

# -------------tkinter-----------------
window = Tk()

# 창을 screen 중간에 열기
w = window.winfo_screenwidth()  # width for the Tk root
h = 800  # height for the Tk root

ws = window.winfo_screenwidth()  # width of the screen
hs = window.winfo_screenheight()  # height of the screen
print(ws, hs)
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)

window.geometry('%dx%d+%d+%d' % (w, h, x, y))
window.minsize(w, h)
window.configure(bg="#FFFFFF")
window.title("캡스톤디자인 NETFLEX팀 주제2")
window.bind('<Escape>', lambda e: window.quit())  # esc로 창 종료
window.resizable(True, True)
header_footerFont = Font(family='Tahoma', size=10)
scaleFont = Font(family='Tahoma', size=12)

# 상단 업로드 버튼
header_frame = Frame(window, bg="white")
header_frame.place(x=0, y=0, width=w, height=30)

upload_img_btn = Button(header_frame, text="upload image",
                        bg="midnightblue",
                        fg="white",
                        width="15",
                        font=scaleFont,
                        command=lambda: predict_all())  # find_directory())
upload_img_btn.pack(side="right")

# Canvas 3x2
canvas_frame = Frame(window, bg="green")
canvas_frame.place(x=0, y=30, width=w, height=h - 60)

# Footer
footer_label = Label(window, bg="#B7472A", font=header_footerFont, fg="white", text="Netflex.t of Sejong Univ.")
footer_label.place(x=0, y=h - 30, width=w, height=30)

# Top layout
frame_tmp00 = Frame(canvas_frame, borderwidth=2, relief="ridge")
frame_tmp01 = Frame(canvas_frame, borderwidth=2, relief="ridge")
frame_tmp10 = Frame(canvas_frame, borderwidth=2, relief="ridge")
frame_tmp11 = Frame(canvas_frame, borderwidth=2, relief="ridge")

label00 = Label(master=frame_tmp00, width=int(w / 2), bg="#B7472A", fg="white", text="Original")
label01 = Label(master=frame_tmp01, width=int(w / 2), bg="#B7472A", fg="white", text="Bounding Box")
label10 = Label(master=frame_tmp10, width=int(w / 2), bg="#B7472A", fg="white", text="Segmentation")
label11 = Label(master=frame_tmp11, width=int(w / 2), bg="#B7472A", fg="white", text="Crack size")

video00 = Label(master=frame_tmp00, width=int(w / 2))
video01 = Label(master=frame_tmp01, width=int(w / 2))
video10 = Label(master=frame_tmp10, width=int(w / 2))
video11 = Label(master=frame_tmp11, width=int(w / 2))

frame_tmp00.place(x=0, y=0, width=int(w / 2), height=370)
frame_tmp01.place(x=int(w / 2), y=0, width=int(w / 2), height=370)
frame_tmp10.place(x=0, y=370, width=int(w / 2), height=370)
frame_tmp11.place(x=int(w / 2), y=370, width=int(w / 2), height=370)

label00.pack(side="top")
label01.pack(side="top")
label10.pack(side="top")
label11.pack(side="top")

video00.pack(side="bottom", fill=BOTH, expand=YES)
video01.pack(side="bottom", fill=BOTH, expand=YES)
video10.pack(side="bottom", fill=BOTH, expand=YES)
video11.pack(side="bottom", fill=BOTH, expand=YES)

pb01 = Progressbar(video01, orient=HORIZONTAL, length=100, mode='determinate')
pb10 = Progressbar(video10, orient=HORIZONTAL, length=100, mode='determinate')
pb11 = Progressbar(video11, orient=HORIZONTAL, length=100, mode='determinate')

# -----------------------------------

if __name__ == "__main__":
    print("start")
    window.mainloop()
    print("end")
