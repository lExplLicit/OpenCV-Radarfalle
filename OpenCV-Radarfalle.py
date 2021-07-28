import cv2
from datetime import datetime
from tracker import *
import requests
import numpy as np
import time
import threading



def uploadFoto(filepath , type="blitzerfoto", speed = 0, netto = 0 ):
    
    uploadkey = '123456789'

    if type == "blitzerfoto":
        url = 'https://node-red.deinedomain.de/blitzerfoto?speed=' + str(speed) + '&netto=' + str(netto) + '&uploadkey=' + uploadkey
        files = {'picture': open(filepath, 'rb').read()}
        requests.post(url, files=files)
        print("Blitzerfoto wurde hochgeladen:                   "+ filepath)

    elif type == "fahrzeugfoto":
        url = 'https://node-red.deinedomain.de/fahrzeugfoto?uploadkey=' + uploadkey
        files = {'picture': open(filepath, 'rb').read()}
        requests.post(url, files=files)
        print("Fahzeugfoto wurde hochgeladen:                   "+ filepath)

    return



def calcSpeed(distance_m, time_s ,tolerance_m):

    if time_s <= 0.01:
        time_s = 0.01

    speed = (distance_m / time_s) * 3.6
    speed = round(speed,2)

    speed_max = ((distance_m + tolerance_m) / time_s) * 3.6
    
    tolerance = (speed_max - speed)
    tolerance = round(tolerance,2)

    netto = speed - tolerance
    netto = round(netto,2)


    
    print( str(distance_m) + "m in "+ str(time_s) +"s -> " + str(speed) + " km/h") 
    
    print("Geschwindigkeit:                                 " + str(speed) + " km/h")

    print("Toleranzbereich:                                 " + str(tolerance) + " km/h")

    print("Geschwindigkeit abz. Toleranz:                   " + str(netto) + " km/h")

    return speed , tolerance , netto


def main():
    print("Starting Blitzer ...")

    min_object_size = 6500 #6000
    max_object_count = 2
    gaussian_blur = 5
    measure_distance = 7.5
    measure_distance_tolerance = 0.8
    algo = "MOG2"
    videofile = False
    upload_picture = False


    if videofile:
        upload_picture = False
        print("Using prerecorded Video ...")
        cap = cv2.VideoCapture('videofile.mp4')
        
    else:
        print("Initializing webcam ...")
        cap = cv2.VideoCapture(0)
        

    cap.set(3, 1280)
    cap.set(4, 720)

    

    if algo == 'MOG2':
        print("Using MOG2 algorithm for movement detection\n")
        background_substractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold=70)
    else:
        print("Using KNN algorithm for movement detection\n")
        background_substractor = cv2.createBackgroundSubtractorKNN()

    tracker = EuclideanDistTracker()
    

    letzte_messung = None
    letzte_messung_speed = None
    letzte_messung_toleranz = None
    
    geblizt_array = []
    geblizte_ids_rechts = []
    geblizte_ids_links = []
    gemessene_ids = []

    prev_frame_time = 0
    new_frame_time = 0

    while True:

        # Read frame and resize
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280,720))

        # Convert frame to gray and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(gaussian_blur,gaussian_blur),0)

        # Set up ignored areas by applying white rectangle
        cv2.rectangle (gray,(0,0),(1280,220),(230,230,230),-1)
        cv2.rectangle (gray,(0,450),(1280,720),(255,255,255),-1)

        # Apply bg filtering
        mask = background_substractor.apply(gray)
        mask = cv2.dilate (mask, None, iterations = 4)

        # Find contours and sort by area size
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # Store timestring for this frame
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%y %H:%M:%S.%f")[:-3]
        dt_string_file = now.strftime("%d.%m.%y_%H.%M.%S.%f")[:-3]
        
        # Put timestamp on frame
        cv2.rectangle (frame,(0,0),(1279,40),(0,0,0),-1)
        cv2.putText(frame, dt_string, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        detections = [] 

        # 1. Detect objects (only one. remove break to detect more)
        concount = 0
        for contour in contours:
            concount += 1
            area = cv2.contourArea(contour)
            
            cv2.drawContours(frame, [contour],-1, (0,0,255),1)
            if area > min_object_size:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append([x, y, w, h])

            if concount >= max_object_count:
                break

        # 2. Track objects
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id

            # right flash
            if x >= 700 and x <= 735 and id not in geblizte_ids_rechts:

                geblizte_ids_rechts.append(id)
                geblizt_array.append([id,now,"right"])

                blitz_img = frame.copy()
                crop_img = blitz_img[y-50:y+h+50, x-50:x+w+50]

                cv2.putText(crop_img, dt_string, (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                cv2.imshow("Blitz Rechts", crop_img)
                print("\nBlitz Rechts. Objekt-ID: " + str(id))

                if not videofile:
                    cv2.imwrite("temp/fahrzeug_foto_rechts.png" , crop_img)


            
            # left flash
            if x >= 150 and x <= 185 and id not in geblizte_ids_links:

                geblizte_ids_links.append(id)
                geblizt_array.append([id, now , "left"])

                blitz_img = frame.copy()
                crop_img = blitz_img[y-50:y+h+50, x-50:x+w+50]
                
                cv2.putText(crop_img, dt_string, (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                cv2.imshow("Blitz Links", crop_img)
                print("\nBlitz Links. Objekt-ID: " + str(id))

                if not videofile:
                    cv2.imwrite("temp/fahrzeug_foto_links.png" , crop_img)
                
            cv2.putText(frame, "ID: " + str(id), (x + 5, y + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # 3 Measure Objects - if id was flashed left and right AND was not measured before
        if id in geblizte_ids_links and id in geblizte_ids_rechts and id not in gemessene_ids:
            
            # Store id so it will not be measured again
            gemessene_ids.append(id)
            print("Das Fahrzeug (" + str(id) + ") wurde zweimal geblizt.")
            
            # Get both measurements for id
            relevante_messungen = []
            for messung in geblizt_array:
                if messung[0] == id:
                    relevante_messungen.append(messung)
                    
            # If id was measured twice, calculate time and speed    
            if len(relevante_messungen) == 2:

                delta = round(datetime.timestamp(relevante_messungen[1][1]) - datetime.timestamp(relevante_messungen[0][1]) , 3)
                speed , toleranz , netto = calcSpeed(measure_distance, delta , measure_distance_tolerance)

                letzte_messung = "ID: " + str(id)+" Time: " + str(round(delta,3 )) + "s Speed: ~"+ str(speed) +" km/h"
                letzte_messung_speed = speed
                letzte_messung_toleranz = toleranz

                use_fahrzeug_foto = ""

                if relevante_messungen[1][2] == "right":
                    print("Das Auto fuhr von Links nach Rechts.             Bergauf" )
                    use_fahrzeug_foto = "temp/fahrzeug_foto_links.png"   
                elif relevante_messungen[1][2] == "left":
                    print("Das Auto fuhr von Rechts nach Links.             Bergab" )
                    use_fahrzeug_foto = "temp/fahrzeug_foto_rechts.png"
            
                print("Erfasste Fahrzeuge (gesamt):                     " + str(gemessene_ids))


                # Generate blitzerfoto
                blitzerfoto = frame.copy()
                cv2.putText(blitzerfoto,  letzte_messung, (570, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
                blitzerfoto = cv2.cvtColor(blitzerfoto, cv2.COLOR_BGR2GRAY)
                blitzerfoto_small = cv2.resize(blitzerfoto, None, fx=0.48,fy=0.48)
                cv2.imshow("Blitzerfoto (komprimiert)", blitzerfoto_small)
                
                # Store pictures if webcam is used
                filename_blitzerfoto = dt_string_file + "_ID" +str(id)+ "_" + str(speed)+"kmh.png"
                filename_blitzerfoto_small = "blitzerfoto_small.png"

                if not videofile:
                    cv2.imwrite("blitzerfotos/" + filename_blitzerfoto, blitzerfoto)
                    cv2.imwrite("temp/" + filename_blitzerfoto_small, blitzerfoto_small)

                # Upload picture if flag is set
                if upload_picture:
                    
                    path = "temp/" + filename_blitzerfoto_small
                    upl_thread_1 = threading.Thread(target=uploadFoto, args=(path, 'blitzerfoto' , speed, netto,))
                    print("Upload gestartet (Blitzerfoto) ...")                 
                    upl_thread_1.start()

                    
                    path = use_fahrzeug_foto
                    upl_thread_2 = threading.Thread(target=uploadFoto, args=(path, 'fahrzeugfoto', ))
                    print("Upload gestartet (Fahrzeugfoto) ...")                 
                    upl_thread_2.start()

        
        # White ignore areas
        cv2.rectangle (frame,(0,0),(1279,220),(255,255,255),1)
        cv2.rectangle (frame,(0,450),(1279,719),(255,255,255),1)

        # Blue detection areas
        cv2.rectangle (frame,(150,220),(185,450),(200,0,0),2)
        cv2.rectangle (frame,(700,220),(735,450),(200,0,0),2)
        

        # Display colored circle if speed is above certain limit
        if letzte_messung is not None:
            cv2.putText(frame, letzte_messung, (570, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            if letzte_messung_speed > 30:
                cv2.circle(frame, (1180,130), 50, (0,0,200), -1)
            else:
                cv2.circle(frame, (1180,130), 50, (0,200,0), -1)

        # Display FPS and thread count
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps_i = int(fps)
        fps = str(fps_i)

        cv2.putText(frame, "F" + fps, (485, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame, "T" + str(threading.active_count()), (430, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        
        # Display frame and mask
        small_mask = cv2.resize(mask, None, fx=0.35,fy=0.35)
        cv2.imshow("Masked Background", small_mask)
        cv2.imshow("VideoCapture", frame)

        
        key = None

        if videofile:
            key = cv2.waitKey(25) 
        else:
            key = cv2.waitKey(1)

        
        if key == 27:
            break

    cap.release()    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()