#================================================================
#  COVID-19 전파 위험도 추정 : 물체 및 장면 감지를 사용한 실시간 화면 분석
#  Estimation of COVID-19 Transmission Risk: Real-time Screen 
#                      Analysis using Object and Scene Detection
#  Team - 최공이지조
#  version 1.0
#================================================================
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import placesCNN
from itertools import combinations


risk_inout = False
def is_close(p1, p2, h1, h2):
    """
    1. Purpose : 두 점 사이의 거리 계산 -> 거리 판단(사회적 거리두기)
    Args:
        p1, p2 = 거리 계산을 위한 두 점
        h1, h2 = 높이
    Returns:
        check = 일정 거리 이상 차이나는지 check 함
    """

    dst = math.sqrt(p1**2 + p2**2)                      # 1. 두 점 사이의 거리
    Aver = (h1+h2)/2                                    # 2. 높이의 평균
    check = False
    
    if dst < Aver :                                     # 두 점 사이 거리 < 높이 평균
        check = True                        
    if max(h1,h2)/min(h1,h2) > 1.5:
        check = False
    return check                                        # 일정 거리 이상 차이나는지 여부


def convertBack(x, y, w, h):
    """
    2. Purpose : 중심 좌표를 직사각형 좌표로 변환
    Args:
        x, y = midpoint of bbox
        w, h = width, height of the bbox
    Returns:
        xmin, ymin, xmax, ymax
    """

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    3.1 Purpose : Person 및 Mask 클래스를 필터링하고 각 탐지에 대한 경계 상자 중심을 가져옴
    Args:
        detections = total detections in one frame
        img = image from detect_image method of darknet
    Returns:
        img with bbox
    """

    if len(detections) > 0:  						    # 프레임에서 감지 여부 확인 (1번 이상)
        centroid_dict = dict() 					    	# person 사전
        mask_good_centroid_dict = dict() 		    	# Mask(Good) 사전
        mask_bad_centroid_dict = dict() 		    	# Mask(Bad)) 사전
        mask_back_none_centroid_dict = dict() 	    	# Mask(None) 사전

        objectId_Person = 0								# person 객체 Count
        objectId_Good = 0								# Mask(Good) 객체 Count
        objectId_Bad = 0								# Mask(Bad) 객체 Count
        objectId_None = 0								# Mask(None) 객체 Count

        # detections 필터링
        for detection in detections:
            name_tag = str(detection[0])                # Coco 파일의 모든 문자열
            # 1. person 태그
            if name_tag == 'person':                    
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	    # 탐지 값 저장
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))                # 중심좌표 -> 직사각형 좌표
                centroid_dict[objectId_Person] = (int(x), int(y), xmin, ymin, xmax, ymax)                   # person 사전
                objectId_Person += 1                    # person 객체 수 Count

            # 2. Mask(good) 태그
            if name_tag == 'good':
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	    # 탐지 값 저장
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))                # 중심좌표 -> 직사각형 좌표
                mask_good_centroid_dict[objectId_Good] = (int(x), int(y), xmin, ymin, xmax, ymax)           # Mask(good) 사전
                objectId_Good += 1                      # Mask(good) 객체 수 Count

            # 3. Mask(bad) 태그
            elif name_tag == 'bad':
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	    # 탐지 값 저장
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))                # 중심좌표 -> 직사각형 좌표
                mask_bad_centroid_dict[objectId_Bad] = (int(x), int(y), xmin, ymin, xmax, ymax)             # Mask(bad) 사전
                objectId_Bad += 1                       # Mask(bad) 객체 수 Count

            # 4. Mask(none) 태그
            elif name_tag == 'back' or name_tag == 'none':
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]             # 탐지 값 저장
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))                # 중심좌표 -> 직사각형 좌표
                mask_back_none_centroid_dict[objectId_None] = (int(x), int(y), xmin, ymin, xmax, ymax, name_tag) # Mask(none) 사전
                objectId_None += 1                      # Mask(none) 객체 수 Count


        """
        3.2 Purpose : 사람들 간 bbox가 서로 가까이 있는지 확인 (사회적 거리두기)
        """
        red_zone_list = []                              # red zone 조건에 있는 객체 ID를 포함하는 리스트
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):                 # 근접 감지의 모든 조합
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]       # 중심 x : 0, y : 1의 차이 확인
            h1, h2 = p1[5] - p1[3], p2[5] - p2[3]       # bounding box 높이 계산
            distanceCheck = is_close(dx, dy, h1, h2) 	# 거리계산(사회적거리두기) : is_close 함수
            if distanceCheck == True:				    # 사회적 거리두기 여부
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)           # red zone 리스트 추가 : id1
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)           # red zone 리스트 추가 : id2
                    red_line_list.append(p2[0:2])

        # 1. person 사전 
        for idx, box in centroid_dict.items():
            if idx in red_zone_list:                    # red zone 리스트 포함 시             
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)      # 빨간색 (red zone)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)      # 연두색
            # Person 단어 출력 - bbox
            font_size = 0.4
            labelSize = cv2.getTextSize('Person', cv2.FONT_HERSHEY_COMPLEX, font_size, 2)
            _x1 = box[2]
            _y1 = box[3]
            _x2 = box[2] + labelSize[0][0]
            _y2 = box[3] - int(labelSize[0][1])
            location = (box[2], box[3])
            cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'Person', location, cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 0), 1)
            
        # 2. Mask(good) 사전
        for idx, box in mask_good_centroid_dict.items():
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 128, 0), 2)          # 초록색
            # Good 단어 출력 - bbox
            font_size = 0.4
            labelSize = cv2.getTextSize('Good', cv2.FONT_HERSHEY_COMPLEX, font_size, 2)
            _x1 = box[2]
            _y1 = box[3]
            _x2 = box[2] + labelSize[0][0]
            _y2 = box[3] - int(labelSize[0][1])
            location = (box[2], box[3])
            cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 128, 0), cv2.FILLED)
            cv2.putText(img, 'Good', location, cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 0), 1)

		# 3. Mask(bad) 사전
        for idx, box in mask_bad_centroid_dict.items():
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (139, 0, 255), 2)        # 보라색
            # Bad 단어 출력 - bbox
            font_size = 0.4
            labelSize = cv2.getTextSize('Bad', cv2.FONT_HERSHEY_COMPLEX, font_size, 2)
            _x1 = box[2]
            _y1 = box[3]
            _x2 = box[2] + labelSize[0][0]
            _y2 = box[3] - int(labelSize[0][1])
            location = (box[2], box[3])
            cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (139, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Bad', location, cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 0), 1)

        # 4. Mask(none) 사전
        for idx, box in mask_back_none_centroid_dict.items():
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 165, 0), 2)        # 주황색
            # none 단어 출력 - bbox
            font_size = 0.4
            labelSize = cv2.getTextSize('None', cv2.FONT_HERSHEY_COMPLEX, 0.4, 2)
            _x1 = box[2]
            _y1 = box[3]
            _x2 = box[2] + labelSize[0][0]
            _y2 = box[3] - int(labelSize[0][1])
            location = (box[2], box[3])
            cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (255, 165, 0), cv2.FILLED)
            cv2.putText(img, 'None', location, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)


        
        """
        3.3 Purpose : Risk 계산 / 출력
        """
    	# 1. 마스크 착용 여부에 따른 위험도
        mask_person = objectId_Good + objectId_Bad      # 마스크 착용한 사람 수
        if mask_person == 0:
            mask_point = 0
        else:
            mask_point = ((objectId_Bad/mask_person) + (objectId_Good/mask_person)*0.15)*100    # 마스크 착용 시 감염 위험률 15% 까지 감소 (질병관리본부)

    	# 2. 군중 밀집에 따른 위험도
        safe_distance_person = objectId_Person - len(red_zone_list)                             # 안전한 거리에 있는 사람 수
        if objectId_Person == 0:
            distance_point = 0
        else:
            distance_point = ((len(red_zone_list)/objectId_Person) + (safe_distance_person/objectId_Person)*0.18)*100       # 안전한 거리에 있을 시 감염 위험률 18% 까지 감소 (질병관리본부)

        # [1, 2]에 따른 위험도 산출
        risk_point = mask_point + distance_point

    	# 3. 장소 - 공간 개폐 여부
        if risk_inout:                                  # 실내에 있으면 위험도 증가
            risk_point *= 1.5
        

        """ 출력 """
        # Risk 값
        text = "Risk Score : %0.2f" % risk_point

        # risk에 따른 등급
        grade_text = "Risk Grade : "
        if risk_point < 60:
            grade_text += "Safe"
        elif 60 <= risk_point < 100:
            grade_text += "Lower Risk(Caution)"
        elif 100 <= risk_point < 140:
            grade_text += "Medium Risk"
        elif 140 <= risk_point < 180:
            grade_text += "High Risk"
        else:
            grade_text += "Very High Risk"

        # Mask 및 distance text
        mask_text = "Good : {0} Bad : {1} None : {2}".format(objectId_Good, objectId_Bad, objectId_None)
        distance_text = "RedP : {0} GreenP : {1}".format(len(red_zone_list), safe_distance_person)
        
        # display - 좌측상단
        location = (10,25)
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        location = (10,60)
        cv2.putText(img, grade_text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        location = (10,95)
        cv2.putText(img, mask_text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        location = (10,130)
        cv2.putText(img, distance_text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
    return img

netMain = None
metaMain = None
altNames = None
def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4-custom-5class.cfg"
    weightPath = "./yolov4-custom-5class_7000.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
        metaMain = [metaMain.names[i].decode("ascii") for i in range(metaMain.classes)]
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    
    cap = cv2.VideoCapture(0) # 캠사용
    # cap = cv2.VideoCapture("./test.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2

    out = cv2.VideoWriter(
            "./cctv_sample_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (new_width, new_height))

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    risk_inout = placesCNN.run_place_detect()
    YOLO()
