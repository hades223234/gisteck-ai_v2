import asyncio
import websockets
import numpy as np
import torch
import cv2
import time
import os
import uuid
import requests
import json
from datetime import datetime
import threading
import aiohttp

API_URL = "http://127.0.0.1:5001/api/"
model = torch.hub.load("yolov5", "custom", path="best.pt", source="local")
model.conf = 0.4  # Set confidence threshold
model.iou = 0.5   # Set IoU threshold for NMS

STAGING_AREA_BOUNDS = [
    (160, 240),  # Top-left  Door Side Line
    (344, 110),  # Top-right Door Side Line
    (580, 260),  # Bottom-right In/Out Side Line
    (458, 638)   # Bottom-left In/Out Side Line
]
STAGING_AREA_TEXT_POS = (
    int((STAGING_AREA_BOUNDS[1][0] + STAGING_AREA_BOUNDS[2][0]) / 2 + 10),
    int((STAGING_AREA_BOUNDS[1][1] + STAGING_AREA_BOUNDS[2][1]) / 2 )
)
DOOR_STATUS = ""
PALLET_STATUS = "Init"
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PALLET = (200, 0, 0)
G_FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_WIDTH = 1
G_FONT_SCALE = 0.5
def sendPostData(doc_cat, doc_type, pallet_id, door_id, comments):
    global API_URL
    if pallet_id is not None and isinstance(pallet_id, uuid.UUID):
        pallet_id = str(pallet_id)

    #Read from customizing.....??? replace with camera IP
    payload = {
        "WHNUM": "WH001",
        "CameraNO": "Camera01",
        "DocCat": doc_cat,       
        "DocType": doc_type,
        "PAK_ID": pallet_id,
        "DoorNO": door_id
    }
    
    # Create a function to handle the API request asynchronously
    async def send_request_async():
        try:
            url = f"{API_URL.rstrip('/')}/{doc_cat}"
            print(f"Sending async request to {url} with payload: {payload}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    # Check if the request was successful
                    if response.status >= 400:
                        error_text = await response.text()
                        print(f"❌ HTTP Error {response.status}: {error_text}")
                        return {"error": f"HTTP Error {response.status}", "details": error_text}
                    
                    # Return the JSON response
                    result = await response.json()
                    print(f"✅ {comments} API Response: {result}")
                    return result
        except aiohttp.ClientConnectorError:
            print(f"❌ Connection Error: Could not connect to {url}")
            return {"error": "Connection Error"}
        except asyncio.TimeoutError:
            print(f"❌ Timeout Error: Request to {url} timed out")
            return {"error": "Timeout Error"}
        except Exception as e:
            print(f"❌ API Request Failed: {e}")
            return {"error": str(e)}
    
    # Create a function to run the async task in a separate thread
    def run_async_in_thread():
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async function in the new loop
            result = loop.run_until_complete(send_request_async())
            return result
        finally:
            loop.close()
    
    # Start a new thread to handle the API request
    thread = threading.Thread(target=run_async_in_thread)
    thread.daemon = True  # Set as daemon so it doesn't block program exit
    thread.start()
def ValidateyStagingArea():
    if (STAGING_AREA_BOUNDS[0][0] > STAGING_AREA_BOUNDS[1][0]):
        return False
    if (STAGING_AREA_BOUNDS[3][0] > STAGING_AREA_BOUNDS[2][0]):
        return False
    return True

def drawStagingArea(frame):
    # Draw the quadrilateral
    # for i in range(len(STAGING_AREA_BOUNDS)):
    #     pt1 = STAGING_AREA_BOUNDS[i]
    #     pt2 = STAGING_AREA_BOUNDS[(i + 1) % len(STAGING_AREA_BOUNDS)]
    #     cv2.line(frame, pt1, pt2, (0, 0, 255), 1)
    points = np.array(STAGING_AREA_BOUNDS, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=COLOR_RED, thickness=LINE_WIDTH)
    
    # Add label for the quadrilateral area
    cv2.putText(frame, "STAGING AREA", STAGING_AREA_TEXT_POS,
                G_FONT, G_FONT_SCALE, COLOR_RED, LINE_WIDTH)
                
    cv2.putText(frame, DOOR_STATUS, (10, 30),
                G_FONT, G_FONT_SCALE, COLOR_GREEN, LINE_WIDTH)

    cv2.putText(frame, PALLET_STATUS, (10, 50),
                G_FONT, G_FONT_SCALE, COLOR_GREEN, LINE_WIDTH)

def showFrame(frame):    
    cv2.imshow('Received Frame', frame)

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def calculate_distance_to_line(point, line_point1, line_point2):
    x, y = point
    x1, y1 = line_point1
    x2, y2 = line_point2
        
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    
    distance = abs(A*x + B*y + C) / ((A**2 + B**2)**0.5)
    return distance
def distance_to_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
def transform_to_position_coordinates(point, origin, x_axis_point):    
    # Calculate the unit vector along the door side line (new x-axis)
    dx = x_axis_point[0] - origin[0]
    dy = x_axis_point[1] - origin[1]
    door_line_length = (dx**2 + dy**2)**0.5
    x_unit_vector = (dx / door_line_length, dy / door_line_length)
    
    # Calculate the unit vector perpendicular to the door side line (new y-axis)
    # Rotate 90 degrees counterclockwise
    y_unit_vector = (-x_unit_vector[1], x_unit_vector[0])
    
    # Translate point relative to the new origin
    translated_x = point[0] - origin[0]
    translated_y = point[1] - origin[1]
    
    # Project onto the new axes
    new_x = translated_x * x_unit_vector[0] + translated_y * x_unit_vector[1]
    new_y = translated_x * y_unit_vector[0] + translated_y * y_unit_vector[1]
    
    return (new_x, new_y)
def getWidthHeight(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    return (width, height)
def getCenterPos(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)
objectDoors = [] # x1, y1, x2, y2, conf, width, heigh, x0, y0. status, validCounts, isValidated, updateTime, isUpdated
def validateDoors():
    global DOOR_STATUS
    #print(f"door status: {DOOR_STATUS}")
    # if isValidated is false and updateTime is over 3 seconds, remove these doors from objectDoors
    current_time = time.time()
    i = 0
    while i < len(objectDoors):
        if not objectDoors[i]["isValidated"] and current_time - objectDoors[i]["updateTime"] > 3:
            objectDoors.pop(i)
        else:
            if objectDoors[i]["validCounts"] > 2:
                objectDoors[i]["validCounts"] = 2
                objectDoors[i]["isValidated"] = True
                if objectDoors[i]["isUpdated"] == False:
                    door_status = "Open" if objectDoors[i]["status"] else "Close"
                    DOOR_STATUS =f"Door{i}: {door_status}"
                    #print(f"---------{i}:{DOOR_STATUS}")
                    sendPostData("Door", door_status, None, f"Door{i}", f"Door {door_status}")
                    objectDoors[i]["isUpdated"] = True
                    break
            i += 1

def updateDoorStatus(doors):
    for door in doors:
        w, h = getWidthHeight(door["x1"], door["y1"], door["x2"], door["y2"])
        x0, y0 = getCenterPos(door["x1"], door["y1"], door["x2"], door["y2"])
        x0, y0 = transform_to_position_coordinates((x0, y0), STAGING_AREA_BOUNDS[0], STAGING_AREA_BOUNDS[1])
        min_distance = 100
        id = -1
        for i, objectDoor in enumerate(objectDoors):
            if min_distance > abs(objectDoor["x0"] - x0):
                min_distance = abs(objectDoor["x0"] - x0)
                id = i
        if id > -1 :
            if min_distance < objectDoors[id]["width"]/2: # exist door
                status = abs(y0) > h
                if objectDoors[id]["status"] == status: # same status
                    objectDoors[id]["x0"] = x0
                    objectDoors[id]["y0"] = y0
                    objectDoors[id]["validCounts"] += 1
                    objectDoors[id]["updateTime"] = time.time()                    
                else : # new status
                    objectDoors[id]["x1"] = door["x1"]
                    objectDoors[id]["y1"] = door["y1"]
                    objectDoors[id]["x2"] = door["x2"]
                    objectDoors[id]["y2"] = door["y2"]
                    objectDoors[id]["conf"] = door["conf"]
                    objectDoors[id]["width"] = w
                    objectDoors[id]["height"] = h
                    objectDoors[id]["x0"] = x0
                    objectDoors[id]["y0"] = y0
                    objectDoors[id]["status"] = status
                    objectDoors[id]["validCounts"] = 0
                    objectDoors[id]["updateTime"] = time.time()
                    objectDoors[id]["isUpdated"] = not objectDoors[id]["isUpdated"]
            else : # new door
                status = abs(y0) > h
                if x0 < objectDoors[id]["x0"]:
                    id -= 1
                objectDoors.insert(id, {"x1":door["x1"], "y1":door["y1"], "x2":door["x2"], "y2":door["y2"], "conf": door["conf"], "width":w, "height":h, "x0":x0, "y0":y0, "status":status, "validCounts": 0, "isValidated": False, "updateTime": time.time(), "isUpdated":True})
        else: # empty door
            status = abs(y0) > h
            objectDoors.append({"x1":door["x1"], "y1":door["y1"], "x2":door["x2"], "y2":door["y2"], "conf": door["conf"], "width":w, "height":h, "x0":x0, "y0":y0, "status":status, "validCounts": 0, "isValidated": False, "updateTime": time.time(), "isUpdated":True})   

def processDoors(doors):
    updateDoorStatus(doors)
    validateDoors()

# Pallet tracking data structure
objectPallets = [] # uuid, x1, y1, x2, y2, conf, width, height, x0, y0, is_inside, first_seen, last_seen, track_history

fixedPallets = [] # uuid, x1, y1, x2, y2, conf, width, height, x0, y0, updateTime, track_history
movingPallets = [] # uuid, x1, y1, x2, y2, conf, width, height, x0, y0, updateTime, track_history
doorSidePallets = [] # uuid, x1, y1, x2, y2, conf, width, height, x0, y0, updateTime
rackSidePallets = [] # uuid, x1, y1, x2, y2, conf, width, height, x0, y0, updateTime

newMovingPallets = []
newDoorSidePallets = []
newRackSidePallets = []
    
def validatePallet(pallet):
    current_time = time.time()
    x1, y1, x2, y2 = pallet["x1"], pallet["y1"], pallet["x2"], pallet["y2"]
    conf = pallet["conf"]
    width, height = getWidthHeight(x1, y1, x2, y2)
    x0, y0 = getCenterPos(x1, y1, x2, y2)
    is_inside = is_point_inside_polygon((x0, y0), STAGING_AREA_BOUNDS)
    xd, yd = transform_to_position_coordinates((x0, y0), STAGING_AREA_BOUNDS[0], STAGING_AREA_BOUNDS[1])
    new_pallet = {
        "uuid": None,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "conf": conf,
        "width": width,
        "height": height,
        "x0": x0,
        "y0": y0,
        "xd": xd,
        "yd": yd,
        "updateTime": current_time
    }
    if is_inside:
        matched = False
        for newPallet in newMovingPallets:
            # Check if pallet is fully contained in newPallet or vice versa
            if ((pallet["x1"] >= newPallet["x1"] and pallet["y1"] >= newPallet["y1"] and 
                 pallet["x2"] <= newPallet["x2"] and pallet["y2"] <= newPallet["y2"]) or 
                (newPallet["x1"] >= pallet["x1"] and newPallet["y1"] >= pallet["y1"] and 
                 newPallet["x2"] <= pallet["x2"] and newPallet["y2"] <= pallet["y2"])):
                if newPallet["conf"] < pallet["conf"]:
                    newPallet.update(new_pallet)
                    # print(f"update newMovingPallets[{len(newMovingPallets)}] === {newPallet}")
                matched = True
                break
        if not matched:
            newMovingPallets.append(new_pallet)
            # print(f"add newMovingPallets[{len(newMovingPallets)}] === {new_pallet}")
    else:
        xr, yr = transform_to_position_coordinates((x0, y0), STAGING_AREA_BOUNDS[1], STAGING_AREA_BOUNDS[2])
        if abs(yr) > abs(yd):
            matched = False
            for newPallet in newDoorSidePallets:
                # Check if pallet is fully contained in newPallet or vice versa
                if ((pallet["x1"] >= newPallet["x1"] and pallet["y1"] >= newPallet["y1"] and 
                     pallet["x2"] <= newPallet["x2"] and pallet["y2"] <= newPallet["y2"]) or 
                    (newPallet["x1"] >= pallet["x1"] and newPallet["y1"] >= pallet["y1"] and 
                     newPallet["x2"] <= pallet["x2"] and newPallet["y2"] <= pallet["y2"])):
                    if newPallet["conf"] < pallet["conf"]:
                        newPallet.update(new_pallet)
                        # print(f"update newDoorSidePallets[{len(newDoorSidePallets)}] === {newPallet}")
                    matched = True
                    break
            if not matched:
                for i, objectDoor in enumerate(objectDoors):
                    if objectDoor["status"] and objectDoor["x1"] < x0 and x0 < objectDoor["x2"]: # opened
                        newDoorSidePallets.append(new_pallet)
                        # print(f"add newDoorSidePallets[{len(newDoorSidePallets)}] === {new_pallet}")
                        break
        else:
            matched = False
            for newPallet in newRackSidePallets:
                # Check if pallet is fully contained in newPallet or vice versa
                if ((pallet["x1"] >= newPallet["x1"] and pallet["y1"] >= newPallet["y1"] and 
                     pallet["x2"] <= newPallet["x2"] and pallet["y2"] <= newPallet["y2"]) or 
                    (newPallet["x1"] >= pallet["x1"] and newPallet["y1"] >= pallet["y1"] and 
                     newPallet["x2"] <= pallet["x2"] and newPallet["y2"] <= pallet["y2"])):
                    if newPallet["conf"] < pallet["conf"]:
                        newPallet.update(new_pallet)
                        # print(f"update newRackSidePallets[{len(newRackSidePallets)}] === {newPallet}")
                    matched = True
                    break
            if not matched:
                newRackSidePallets.append(new_pallet)
                # print(f"add newRackSidePallets[{len(newRackSidePallets)}] === {new_pallet}")

def resetNewPallets():
    global newMovingPallets, newDoorSidePallets, newRackSidePallets
    newMovingPallets = []
    newDoorSidePallets = []
    newRackSidePallets = []
def resetObjects():
    global fixedPallets, movingPallets, doorSidePallets, rackSidePallets, objectDoors, DOOR_STATUS, PALLET_STATUS
    fixedPallets = [] 
    movingPallets = [] 
    doorSidePallets = [] 
    rackSidePallets = []
    objectDoors = []
    resetNewPallets()
    DOOR_STATUS = "Init"
    PALLET_STATUS = "Reset"

def processPallets():
    global PALLET_STATUS, newMovingPallets, newDoorSidePallets, newRackSidePallets
    current_time = time.time()
    
    # Process door side pallets
    i = 0
    while i < len(doorSidePallets):
        min_distance = 50
        id  = -1
        for j, pallet in enumerate(newDoorSidePallets):
            distance = distance_to_points((pallet["x0"], pallet["y0"]), (doorSidePallets[i]["x0"], doorSidePallets[i]["y0"]))
            if min_distance > distance:
                min_distance = distance
                id  = j        
        if id > -1: # same door side state
            pallet_obj = doorSidePallets[i]
            pallet_obj["x1"] = newDoorSidePallets[id]["x1"]
            pallet_obj["y1"] = newDoorSidePallets[id]["y1"]
            pallet_obj["x2"] = newDoorSidePallets[id]["x2"]
            pallet_obj["y2"] = newDoorSidePallets[id]["y2"]
            pallet_obj["conf"] = newDoorSidePallets[id]["conf"]
            pallet_obj["width"] = newDoorSidePallets[id]["width"]
            pallet_obj["height"] = newDoorSidePallets[id]["height"]
            pallet_obj["x0"] = newDoorSidePallets[id]["x0"]
            pallet_obj["y0"] = newDoorSidePallets[id]["y0"]
            pallet_obj["xd"] = newDoorSidePallets[id]["xd"]
            pallet_obj["yd"] = newDoorSidePallets[id]["yd"]
            pallet_obj["updateTime"] = newDoorSidePallets[id]["updateTime"]
            newDoorSidePallets.pop(id)        
            # print(f"update doorSidePallets[{len(doorSidePallets)}] === {pallet_obj}")    
        else:            
            for j, pallet in enumerate(newMovingPallets):
                distance = distance_to_points((pallet["x0"], pallet["y0"]), (doorSidePallets[i]["x0"], doorSidePallets[i]["y0"]))
                if min_distance > distance:
                    min_distance = distance
                    id  = j  
            if id > -1 and doorSidePallets[i]["uuid"] is None: # unload state
                p_uuid = uuid.uuid4()
                sendPostData("Dock", "UOD", p_uuid, None, f"Dock Unload")
                PALLET_STATUS = "unload state"
                print(f"========={PALLET_STATUS}")
                new_pallet = {
                    "uuid": p_uuid,
                    "x1": newMovingPallets[id]["x1"],
                    "y1": newMovingPallets[id]["y1"],
                    "x2": newMovingPallets[id]["x2"],
                    "y2": newMovingPallets[id]["y2"],
                    "conf": newMovingPallets[id]["conf"],
                    "width": newMovingPallets[id]["width"],
                    "height": newMovingPallets[id]["height"],
                    "x0": newMovingPallets[id]["x0"],
                    "y0": newMovingPallets[id]["y0"],
                    "xd": newMovingPallets[id]["xd"],
                    "yd": newMovingPallets[id]["yd"],
                    "updateTime": newMovingPallets[id]["updateTime"],
                    "track_history": [(doorSidePallets[i]["x0"], doorSidePallets[i]["y0"])]
                }
                # print(f"add movingPallets[{len(movingPallets)}] === {new_pallet}")
                movingPallets.append(new_pallet)
                doorSidePallets.pop(i)
                newMovingPallets.pop(id)
            elif current_time - doorSidePallets[i]["updateTime"] > 3: # if update time is over 3 seconds, deleted
                doorSidePallets.pop(i)
            else: # other state
                i += 1
    # Process rack side pallets
    i = 0
    while i < len(rackSidePallets):
        min_distance = 50
        id  = -1
        for j, pallet in enumerate(newRackSidePallets):
            distance = distance_to_points((pallet["x0"], pallet["y0"]), (rackSidePallets[i]["x0"], rackSidePallets[i]["y0"]))
            if min_distance > distance:
                min_distance = distance
                id  = j        
        if id > -1: # same rack side state
            pallet_obj = rackSidePallets[i]
            pallet_obj["x1"] = newRackSidePallets[id]["x1"]
            pallet_obj["y1"] = newRackSidePallets[id]["y1"]
            pallet_obj["x2"] = newRackSidePallets[id]["x2"]
            pallet_obj["y2"] = newRackSidePallets[id]["y2"]
            pallet_obj["conf"] = newRackSidePallets[id]["conf"]
            pallet_obj["width"] = newRackSidePallets[id]["width"]
            pallet_obj["height"] = newRackSidePallets[id]["height"]
            pallet_obj["x0"] = newRackSidePallets[id]["x0"]
            pallet_obj["y0"] = newRackSidePallets[id]["y0"]
            pallet_obj["xd"] = newRackSidePallets[id]["xd"]
            pallet_obj["yd"] = newRackSidePallets[id]["yd"]
            pallet_obj["updateTime"] = newRackSidePallets[id]["updateTime"]
            newRackSidePallets.pop(id)     
            # print(f"update rackSidePallets[{len(rackSidePallets)}] === {pallet_obj}")       
        else:            
            for j, pallet in enumerate(newMovingPallets):
                distance = distance_to_points((pallet["x0"], pallet["y0"]), (rackSidePallets[i]["x0"], rackSidePallets[i]["y0"]))
                if min_distance > distance:
                    min_distance = distance
                    id  = j  
            if id > -1 and rackSidePallets[i]["uuid"] is None: # in state
                p_uuid = uuid.uuid4()
                sendPostData("Stage", "IN", p_uuid, None, f"Stage In")
                PALLET_STATUS = "in state"
                print(f"========={PALLET_STATUS}")
                new_pallet = {
                    "uuid": p_uuid,
                    "x1": newMovingPallets[id]["x1"],
                    "y1": newMovingPallets[id]["y1"],
                    "x2": newMovingPallets[id]["x2"],
                    "y2": newMovingPallets[id]["y2"],
                    "conf": newMovingPallets[id]["conf"],
                    "width": newMovingPallets[id]["width"],
                    "height": newMovingPallets[id]["height"],
                    "x0": newMovingPallets[id]["x0"],
                    "y0": newMovingPallets[id]["y0"],
                    "xd": newMovingPallets[id]["xd"],
                    "yd": newMovingPallets[id]["yd"],
                    "updateTime": newMovingPallets[id]["updateTime"],
                    "track_history": [(rackSidePallets[i]["x0"], rackSidePallets[i]["y0"])]
                }
                # print(f"add movingPallets[{len(movingPallets)}] === {new_pallet}")
                movingPallets.append(new_pallet)
                rackSidePallets.pop(i)
                newMovingPallets.pop(id)
            elif current_time - rackSidePallets[i]["updateTime"] > 3: # if update time is over 3 seconds, deleted
                rackSidePallets.pop(i)
            else: # other state
                i += 1
    # Process moving pallets
    i = 0
    while i < len(movingPallets):
        if movingPallets[i]["uuid"] is None and len(movingPallets[i]["track_history"])>5:
            moving_id = -1
            min_distance = 1000
            for j, pallet in enumerate(movingPallets):
                if movingPallets[j]["uuid"] is None or current_time - movingPallets[j]["updateTime"] < 1 :
                    continue
                distance = distance_to_points((movingPallets[j]["x0"], movingPallets[j]["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
                if min_distance > distance:
                    min_distance = distance
                    moving_id  = j 
            if moving_id > -1:
                #print(f"===========delete movingPallets[{len(movingPallets)}] === {i}, {moving_id}")
                #print(f"===========infos [{movingPallets[i]}] === {movingPallets[moving_id]}")
                movingPallets[i]["uuid"] = movingPallets[moving_id]["uuid"]
                movingPallets.pop(moving_id)
                if moving_id < i:
                    i -=1
               
        min_distance = 50
        id  = -1
        for j, pallet in enumerate(newDoorSidePallets):
            distance = distance_to_points((pallet["x0"], pallet["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
            if min_distance > distance:
                min_distance = distance
                id  = j   
        if id > -1: # load state
            if movingPallets[i]["uuid"] is None:
                moving_id = -1
                min_distance = 1000
                for j, pallet in enumerate(movingPallets):
                    if movingPallets[j]["uuid"] is None:
                        continue
                    distance = distance_to_points((movingPallets[j]["x0"], movingPallets[j]["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
                    if min_distance > distance:
                        min_distance = distance
                        moving_id  = j 
                if  moving_id > -1:
                    movingPallets[i]["uuid"] = movingPallets[moving_id]["uuid"]
                    movingPallets.pop(moving_id)
                    if i > moving_id:
                        i -= 1  # Adjust index if we removed an element before current position
            if movingPallets[i]["uuid"] is not None:
                sendPostData("Dock", "LOD", movingPallets[i]["uuid"], None, f"Dock Load")
                PALLET_STATUS = "load state"
                print(f"========={PALLET_STATUS}")
                new_pallet = {
                    "uuid": movingPallets[i]["uuid"],
                    "x1": newDoorSidePallets[id]["x1"],
                    "y1": newDoorSidePallets[id]["y1"],
                    "x2": newDoorSidePallets[id]["x2"],
                    "y2": newDoorSidePallets[id]["y2"],
                    "conf": newDoorSidePallets[id]["conf"],
                    "width": newDoorSidePallets[id]["width"],
                    "height": newDoorSidePallets[id]["height"],
                    "x0": newDoorSidePallets[id]["x0"],
                    "y0": newDoorSidePallets[id]["y0"],
                    "xd": newDoorSidePallets[id]["xd"],
                    "yd": newDoorSidePallets[id]["yd"],
                    "updateTime": newDoorSidePallets[id]["updateTime"]
                }
                doorSidePallets.append(new_pallet)
                movingPallets.pop(i)
                newDoorSidePallets.pop(id)
            else:
                i += 1
        else:    
            for j, pallet in enumerate(newRackSidePallets):
                distance = distance_to_points((pallet["x0"], pallet["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
                if min_distance > distance:
                    min_distance = distance
                    id  = j   
            if id > -1: # out state
                if movingPallets[i]["uuid"] is None:
                    moving_id = -1
                    min_distance = 1000
                    for j, pallet in enumerate(movingPallets):
                        if movingPallets[j]["uuid"] is None:
                            continue
                        distance = distance_to_points((movingPallets[j]["x0"], movingPallets[j]["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
                        if min_distance > distance:
                            min_distance = distance
                            moving_id  = j 
                    if  moving_id > -1:
                        movingPallets[i]["uuid"] = movingPallets[moving_id]["uuid"]
                        movingPallets.pop(moving_id)
                        if i > moving_id:
                            i -= 1  # Adjust index if we removed an element before current position
                if movingPallets[i]["uuid"] is not None:
                    sendPostData("Stage", "Out", movingPallets[i]["uuid"], None, f"Stage Out")
                    PALLET_STATUS = "out state"
                    print(f"========={PALLET_STATUS}")
                    new_pallet = {
                        "uuid": movingPallets[i]["uuid"],
                        "x1": newRackSidePallets[id]["x1"],
                        "y1": newRackSidePallets[id]["y1"],
                        "x2": newRackSidePallets[id]["x2"],
                        "y2": newRackSidePallets[id]["y2"],
                        "conf": newRackSidePallets[id]["conf"],
                        "width": newRackSidePallets[id]["width"],
                        "height": newRackSidePallets[id]["height"],
                        "x0": newRackSidePallets[id]["x0"],
                        "y0": newRackSidePallets[id]["y0"],
                        "xd": newRackSidePallets[id]["xd"],
                        "yd": newRackSidePallets[id]["yd"],
                        "updateTime": newRackSidePallets[id]["updateTime"]
                    }
                    rackSidePallets.append(new_pallet)
                    movingPallets.pop(i)
                    newRackSidePallets.pop(id)
                else:
                    i += 1
            else: 
                for j, pallet in enumerate(newMovingPallets):
                    distance = distance_to_points((pallet["x0"], pallet["y0"]), (movingPallets[i]["x0"], movingPallets[i]["y0"]))
                    if min_distance > distance:
                        min_distance = distance
                        id  = j   
                if id > -1: # moving state
                    pallet_obj = movingPallets[i]
                    pallet_obj["x1"] = newMovingPallets[id]["x1"]
                    pallet_obj["y1"] = newMovingPallets[id]["y1"]
                    pallet_obj["x2"] = newMovingPallets[id]["x2"]
                    pallet_obj["y2"] = newMovingPallets[id]["y2"]
                    pallet_obj["conf"] = newMovingPallets[id]["conf"]
                    pallet_obj["width"] = newMovingPallets[id]["width"]
                    pallet_obj["height"] = newMovingPallets[id]["height"]
                    pallet_obj["x0"] = newMovingPallets[id]["x0"]
                    pallet_obj["y0"] = newMovingPallets[id]["y0"]
                    pallet_obj["xd"] = newMovingPallets[id]["xd"]
                    pallet_obj["yd"] = newMovingPallets[id]["yd"]
                    pallet_obj["updateTime"] = newMovingPallets[id]["updateTime"]
                    pallet_obj["track_history"].append((newMovingPallets[id]["x0"], newMovingPallets[id]["y0"]))
                    if len(pallet_obj["track_history"]) > 10:
                        pallet_obj["track_history"] = pallet_obj["track_history"][-10:]
                        max_deviation = distance_to_points(pallet_obj["track_history"][1], pallet_obj["track_history"][0])
                        for point in pallet_obj["track_history"][2:-1]:
                            if max_deviation < distance_to_points(point, pallet_obj["track_history"][0]):
                                max_deviation = distance_to_points(point, pallet_obj["track_history"][0])
                        if max_deviation < 5:
                            # if rectangle of this pallet_obj is not full contained in all fixedPallets
                            new_pallet = {
                                "uuid": pallet_obj["uuid"],
                                "x1": pallet_obj["x1"],
                                "y1": pallet_obj["y1"],
                                "x2": pallet_obj["x2"],
                                "y2": pallet_obj["y2"],
                                "conf": pallet_obj["conf"],
                                "width": pallet_obj["width"],
                                "height": pallet_obj["height"],
                                "x0": pallet_obj["x0"],
                                "y0": pallet_obj["y0"],
                                "xd": pallet_obj["xd"],
                                "yd": pallet_obj["yd"],
                                "updateTime": pallet_obj["updateTime"],
                                "track_history": pallet_obj["track_history"].copy()
                            }                            
                            matched_fixed_id = -1
                            for j, pallet in enumerate(fixedPallets):
                                if distance_to_points((pallet["x0"], pallet["y0"]), (new_pallet["x0"], new_pallet["y0"])) < 5:
                                    matched_fixed_id = j
                            if matched_fixed_id > -1:
                                new_pallet["uuid"] = fixedPallets[matched_fixed_id]["uuid"]
                                fixedPallets[matched_fixed_id].update(new_pallet)
                                movingPallets.pop(i)
                                i -= 1
                            else:
                                if new_pallet["uuid"]:
                                    fixedPallets.append(new_pallet)
                                    #print(f"------add fixedPallets[{len(fixedPallets)}] === {new_pallet}")
                                    movingPallets.pop(i)
                                    i -= 1
                    newMovingPallets.pop(id) 
                    i += 1
                else:    
                    if current_time - movingPallets[i]["updateTime"] > 3 and movingPallets[i]["uuid"] is None:
                        movingPallets.pop(i)
                    else:
                        i += 1
    # Process fixed pallets
    i = 0
    while i < len(fixedPallets):
        min_distance = 5
        id  = -1
        for j, pallet in enumerate(newMovingPallets):
            distance = distance_to_points((pallet["x0"], pallet["y0"]), (fixedPallets[i]["x0"], fixedPallets[i]["y0"]))
            if min_distance > distance:
                min_distance = distance
                id  = j  
        if id > -1:
            if newMovingPallets[id]["conf"] > 0.5:
                fixedPallets[i]["x1"] = newMovingPallets[id]["x1"]
                fixedPallets[i]["y1"] = newMovingPallets[id]["y1"]
                fixedPallets[i]["x2"] = newMovingPallets[id]["x2"]
                fixedPallets[i]["y2"] = newMovingPallets[id]["y2"]
                fixedPallets[i]["conf"] = newMovingPallets[id]["conf"]
                fixedPallets[i]["width"] = newMovingPallets[id]["width"]
                fixedPallets[i]["height"] = newMovingPallets[id]["height"]
                fixedPallets[i]["x0"] = newMovingPallets[id]["x0"]
                fixedPallets[i]["y0"] = newMovingPallets[id]["y0"]
                fixedPallets[i]["xd"] = newMovingPallets[id]["xd"]
                fixedPallets[i]["yd"] = newMovingPallets[id]["yd"]
                fixedPallets[i]["updateTime"] = newMovingPallets[id]["updateTime"]
                fixedPallets[i]["track_history"].append((newMovingPallets[id]["x0"], newMovingPallets[id]["y0"]))
                if len(fixedPallets[i]["track_history"]) > 10:
                    fixedPallets[i]["track_history"] = fixedPallets[i]["track_history"][-10:]
            newMovingPallets.pop(id)
        min_distance = 50
        id  = -1
        for j, pallet in enumerate(movingPallets):
            distance = distance_to_points((pallet["x0"], pallet["y0"]), (fixedPallets[i]["x0"], fixedPallets[i]["y0"]))
            if min_distance > distance:
                min_distance = distance
                id  = j        
        if id > -1 and min_distance > 30 and movingPallets[id]["uuid"] is None: # moving state
            #print(f"moving state[{len(movingPallets)}] === {fixedPallets[i]["uuid"]}")
            movingPallets[id]["uuid"] = fixedPallets[i]["uuid"]
            fixedPallets.pop(i)            
        else:       
            i += 1     
            
    # Add new moving pallets
    for pallet in newMovingPallets:
        new_pallet = {
            "uuid": None,
            "x1": pallet["x1"],
            "y1": pallet["y1"],
            "x2": pallet["x2"],
            "y2": pallet["y2"],
            "conf": pallet["conf"],
            "width": pallet["width"],
            "height": pallet["height"],
            "x0": pallet["x0"],
            "y0": pallet["y0"],
            "xd": pallet["xd"],
            "yd": pallet["yd"],
            "updateTime": pallet["updateTime"],
            "track_history": [(pallet["x0"], pallet["y0"])]
        }
        movingPallets.append(new_pallet)
        #print(f"add movingPallets[{len(movingPallets)}] === {new_pallet}")
    for pallet in newDoorSidePallets:
        new_pallet = {
            "uuid": None,
            "x1": pallet["x1"],
            "y1": pallet["y1"],
            "x2": pallet["x2"],
            "y2": pallet["y2"],
            "conf": pallet["conf"],
            "width": pallet["width"],
            "height": pallet["height"],
            "x0": pallet["x0"],
            "y0": pallet["y0"],
            "xd": pallet["xd"],
            "yd": pallet["yd"],
            "updateTime": pallet["updateTime"]
        }
        doorSidePallets.append(new_pallet)
        #print(f"add doorSidePallets[{len(doorSidePallets)}] === {new_pallet}")
    for pallet in newRackSidePallets:
        new_pallet = {
            "uuid": None,
            "x1": pallet["x1"],
            "y1": pallet["y1"],
            "x2": pallet["x2"],
            "y2": pallet["y2"],
            "conf": pallet["conf"],
            "width": pallet["width"],
            "height": pallet["height"],
            "x0": pallet["x0"],
            "y0": pallet["y0"],
            "xd": pallet["xd"],
            "yd": pallet["yd"],
            "updateTime": pallet["updateTime"]
        }
        rackSidePallets.append(new_pallet)
        #print(f"add rackSidePallets[{len(rackSidePallets)}] === {new_pallet}")
    resetNewPallets()

objectForklift = None # x1, y1, x2, y2, conf, width, height, center_x, center_y
def processForklift(forklift):
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (190, 0, 190), 2)
    w, h = getWidthHeight(forklift["x1"], forklift["y1"], forklift["x2"], forklift["y2"])
    x0, y0 = getCenterPos(forklift["x1"], forklift["y1"], forklift["x2"], forklift["y2"])
    objectForklift = {"x1":forklift["x1"], "y1":forklift["y1"], "x2":forklift["x2"], "y2":forklift["y2"], "conf": forklift["conf"], "width":w, "height":h, "x0":x0, "y0":y0}
    pass
def detectObject(frame):
    #print(f"-----------------------------------------------------")
    #pallets = []
    doors = []
    forklift = None
    results = model(frame)
    detections = results.pandas().xyxy[0]
    for _, row in detections.iterrows():    
        x1, y1, x2, y2, conf, class_id, class_name = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
            row["confidence"],
            int(row["class"]),
            row["name"],
        )        
        if class_name == "pallet":
            validatePallet({"x1":x1, "y1":y1, "x2":x2, "y2":y2, "conf":conf})
            #pallets.append({"x1":x1, "y1":y1, "x2":x2, "y2":y2, "conf":conf})
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PALLET, LINE_WIDTH)
        elif class_name == "forklift":
            forklift = {"x1":x1, "y1":y1, "x2":x2, "y2":y2, "conf":conf}
        elif class_name == "door":
            doors.append({"x1":x1, "y1":y1, "x2":x2, "y2":y2, "conf":conf})
            #cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, LINE_WIDTH)
        else:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BLACK, LINE_WIDTH)
    processDoors(doors)
    processPallets()
    # processForklift(forklift)
    
last_save_time = time.time()
frame_counter = 0     
def saveFrame(frame):
    global last_save_time, frame_counter
    current_time = time.time()
    if current_time - last_save_time >= 2:
        save_path = "image"
        os.makedirs(save_path, exist_ok=True) 
        file_name = os.path.join(save_path, f"frame_{frame_counter}.jpg")
        cv2.imwrite(file_name, frame)
        print(f"Saved {file_name}")
        frame_counter += 1
        last_save_time = current_time

# async def video_stream(websocket):
#     try:         
#         processing_flag = 0
#         print("Client connected.")      
#         async for message in websocket:
#             if processing_flag == 1:
#                 print("Warning: Processing...")
#                 continue
#             nparr = np.frombuffer(message, np.uint8)
#             # Decode the image
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decodes as BGR           

#             if frame is None:
#                 print("Warning: Received an empty or corrupt frame.")
#                 continue      
#             print(f"===1== {frame.shape} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
#             #saveFrame(frame)    
#             frame = cv2.resize(frame, (640, 640))
#             # detectObject(frame)   
#             # drawStagingArea(frame)
#             # showFrame(frame)  
#             processing_flag = 1
#             await frame_processor(frame)
#             processing_flag = 0
#             print(f"===2== {frame.shape} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break          
#     except websockets.exceptions.ConnectionClosed:
#         print("Client disconnected.")
#     finally:
#         cv2.destroyAllWindows()
#         resetObjects()
#         print("Websocket signal is 'no', objects reset!")
#         await asyncio.sleep(1)  # Non-blocking sleep

# async def main():
#     start_server = websockets.serve(video_stream, "localhost", 5001)
#     async with start_server:
#         print("WebSocket server started on ws://localhost:5001")
#         await asyncio.Future()
frame_queue = asyncio.Queue(maxsize=1)  # Keep only latest frame
show_frame = None

async def frame_processor():
    global show_frame
    while True:
        frame = await frame_queue.get()
        try:
            # Simulate processing delay
            #print(f"Processing Frame Started: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            await asyncio.to_thread(detectObject, frame)
            #print(f"Processing Frame detectObject end: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            await asyncio.to_thread(drawStagingArea, frame)
            #print(f"Processing Frame drawStagingArea end: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            show_frame = frame
            #await asyncio.to_thread(showFrame, frame)
            #print(f"Processed Frame: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        except Exception as e:
            print(f"Error[frame_processor]: {e}")
        finally:
            frame_queue.task_done()
async def video_stream(websocket):
    global show_frame
    # if websocket.request.path != '/unity1':
    #     return 
    print("Client connected.")
    resetObjects()
    try:
        async for message in websocket:
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Warning: Received empty or corrupt frame.")
                continue

            frame = cv2.resize(frame, (640, 640))
            #print(f"Received Frame: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            if frame_queue.empty():
                await frame_queue.put(frame)
            if show_frame is not None:
                showFrame(show_frame)
            #print(f"Show Frame: {frame.shape} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cv2.destroyAllWindows()
        print("Websocket signal is 'no', objects reset!")
        await asyncio.sleep(1)

async def main():
    start_server = websockets.serve(video_stream, "0.0.0.0", 5000)
    asyncio.create_task(frame_processor())  # Start frame processor in background
    async with start_server:
        print("WebSocket server started on ws://0.0.0.0:5000")
        await asyncio.Future()  # Run forever
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
