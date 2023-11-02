import argparse
import time
from pathlib import Path

import cv2
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from flask import Response
from flask import Flask
from flask import render_template
import threading

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = s.getsockname()[0]
print(ip_address)

import paho.mqtt.client as mqtt
import requests
import json

def on_message(client, userdata, message):
    print("li")
    data1 =[]
    receivedstring = str(message.payload.decode("utf-8"))
    data1=receivedstring.split(",")
    # print(data1)
    if data1[0] == '*':
        print('WEAPON detected and the Alarm is ON')
    if data1[0] == '$':
        print('WEAPON detected and the GATE is Closed')
    with open('config.json', 'w') as json_file:
        json.dump(data1, json_file)

broker_address="broker.hivemq.com"
client = mqtt.Client("WEAPON") 
client.connect(broker_address) 
client.on_message=on_message 
client.subscribe("WEAPON-NT")
response = 0 

with open('config.json') as f:
    data = json.load(f)


serverToken = 'AAAA9myXIrY:APA91bHsrnPMm8TUAk7lfPCdXzdTZdz0riWCQlaoTpWVTCGMiWakwWWEoAFdERvk6LQ8esGb-rRJvFTTY9NRTcVc-O9WrNS_MaE3GNmoJ7tbRrqt46RRXlzIPVO4NBo21LFEA8lRMtSD'
deviceToken = data[0]
headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
      }

body = {
          'notification': {'title': 'Sending push form python script',
                            'body': 'New Message'
                            },
          'to':
              deviceToken,
          'priority': 'high',
        }

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def web_stream(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    opt = pd.Series()
    opt.weights = 'best.pt'
    opt.source = '0'
    opt.img_size = 640
    opt.conf_thres = 0.45
    opt.iou_thres = 0.45
    opt.device = 'cpu'
    opt.view_img = True
    opt.save_txt = False
    opt.save_conf = False
    opt.nosave = True
    opt.classes = None
    opt.agnostic_nms = True
    opt.augment = True
    opt.project = 'runs/detect'
    opt.name = 'exp'
    opt.exist_ok = True
    opt.no_trace = True
    # print(opt)

    # with torch.no_grad():
    #         detect()

    # global notification_flag
    notification_flag = 0
    normalflag = 0

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = True

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        client.loop_start()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if 'knife' in s or 'pistol' in s:
                        if notification_flag == 0:
                            normalflag=0
                            client.publish("WEAPON-ST","1"+","+str(ip_address))
                            with open('config.json') as f:
                                data = json.load(f)
                            serverToken = 'AAAA9myXIrY:APA91bHsrnPMm8TUAk7lfPCdXzdTZdz0riWCQlaoTpWVTCGMiWakwWWEoAFdERvk6LQ8esGb-rRJvFTTY9NRTcVc-O9WrNS_MaE3GNmoJ7tbRrqt46RRXlzIPVO4NBo21LFEA8lRMtSD'
                            deviceToken = data[0]
                            headers = {
                                    'Content-Type': 'application/json',
                                    'Authorization': 'key=' + serverToken,
                                }

                            body = {
                                    'notification': {'title': 'Sending push form python script',
                                                        'body': 'New Message'
                                                        },
                                    'to':
                                        deviceToken,
                                    'priority': 'high',
                                    }
                            response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
                            print(response.status_code)
                            print(response.json())
                            notification_flag=1
                    else:
                        client.publish("WEAPON-ST","0")
                        notification_flag=0
                        normalflag=1

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            with lock:
                outputFrame = im0.copy()
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

       
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
        
@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

    # start a thread that will perform motion detection
    t = threading.Thread(target=web_stream, args=(32,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="0.0.0.0", port="8000", threaded=True, use_reloader=False)
