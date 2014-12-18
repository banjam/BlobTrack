#TODO: 
# Add multithreading?
# Consistency between runs: wait till blobs detected get to a certain number?
# and implement some kind of check later on in the algorithm?
# Add area in blob tracking and write to file
# Look into rounding float to int

# Separate blob detection from tracking
# Write blob positions to files fromd data.txt
# Use json for data saving
# Implement a delay to get a fixed framerate?

import numpy as np
import math
import cv2
import copy
import time
import multiprocessing as mp

# Constants for masking frame
MASK_X1 = 75
MASK_X2 = 472
MASK_Y1 = 180
MASK_Y2 = 585
    
# Parameters for optical flow tracking
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Main class: processes images using Lucas-Kanade
class Lk:
    # Define important variables
    def __init__(self, folder):
        self.track_len = 2
        self.detect_interval = 5
        self.tracks = []
        self.folder = folder
        self.frame_idx = 1
        self.framerate_total = 0
        self.frametime_total = 0
        self.delayed = False

    def find_blobs(self, frame):  
        blobs = []      
        # Find contours and filter for size
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 70 and area > 8:
                blobs.append(c)
        return blobs

    def filterFrame(self, frame):  
        # Select ROI
        frame_ROI = np.zeros_like(frame)
        frame_ROI[MASK_X1:MASK_X2,MASK_Y1:MASK_Y2] = frame[MASK_X1:MASK_X2,MASK_Y1:MASK_Y2]
        
        # Blur and convert frame to gray
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)

        # Threshold
        #ret, frame_thresh = cv2.threshold(frame_gray, 150, 255, 0)         
        frame_adThresh = cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,21,-10)

        # Filter noise
        kernel = np.ones((2,2),np.uint8)
        frame_erode = cv2.erode(frame_adThresh,kernel,iterations = 2)
        frame_dilate = cv2.dilate(frame_erode,kernel,iterations = 1)
            
        return frame_dilate

    # Main loop
    def run(self):
        #pool = mp.Pool(processes = 4)
        fdata = open('data.txt', 'w')
        
        while True:
            start = cv2.getTickCount()

            # Get image to be processed
            frame = cv2.imread(self.folder + "/out" + str(self.frame_idx) + ".jpg")

            # If image doesn't exist, first delay and try again, then quit program
            if frame is None and not self.delayed:
                print "No image found - delaying 200 ms ..."
                self.delayed = True
                time.sleep(0.2)
                continue
            if frame is None and self.delayed:
                print "No image found - quitting"
                cv2.waitKey(0)
                break
            if frame is not None and self.delayed:
                self.delayed = False

            # Pre-process and threshold frame
            curr_frame = self.filterFrame(frame);      

            # For first few frames, find blobs and save to self.tracks
            if self.frame_idx <= 2:
                # Create mask around blobs already found to avoid duplication
                mask = np.zeros_like(curr_frame)
                mask[:] = 255
                #for x,y,area in [np.int32(tr[-1]) for tr in self.tracks]:
                for x,y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x,y), 5, 0, -1)
                frame_mask = cv2.bitwise_and(curr_frame,curr_frame,mask=mask)

                # Find blobs
                blobs = self.find_blobs(frame_mask)

                # Save blobs center position (and area)
                if blobs is not None:
                    for b in blobs:
                        M = cv2.moments(b)
                        x = np.float32(M['m10']/M['m00'])
                        y = np.float32(M['m01']/M['m00'])
                        #area = cv2.contourArea(b)
                        self.tracks.append([(x,y)])
                        #self.tracks.append([(x,y,area)])
                        
            # After first few frames, track the blobs that have been detected           
            if self.frame_idx> 2 & len(self.tracks) > 0:
                img0, img1 = self.prev_frame, curr_frame

                # Lucas-Kanade with back-tracking
                #p0 = np.float32([[tr[-1][0], tr[-1][1]]for tr in self.tracks]).reshape(-1, 1, 2)
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                for i in xrange(1, len(self.tracks)):
                    if not good[i]:
                        self.tracks[i].append((self.tracks[i][-1][0],self.tracks[i][-1][1]))
                        #tr.append((tr[-1][0], tr[-1][1], tr[-1][2]))
                    elif good[i]:
                        self.tracks[i].append(p1.reshape(-1, 2)[i])
                        #tr.append((x, y, tr[-1][2]))
                    if len(self.tracks[i]) > self.track_len:
                        del self.tracks[i][0]

                for i in xrange(1,len(self.tracks)):
                    fdata.write(str(self.tracks[i][-1][0]) + " " + str(self.tracks[i][-1][1]) + "\n")
                fdata.write("----------------------------------------------------------------\n")
    
            # Set previous frames to current
            self.frame_idx += 1
            self.prev_frame = curr_frame
            #self.prev_thresh = frame_thresh

            frame_blobs = copy.copy(curr_frame)
            #frame_blobs = np.zeros_like(curr_frame)
            frame_blobs = cv2.cvtColor(frame_blobs,cv2.COLOR_GRAY2BGR)

            # Draw blobs
            #for x,y,area in [tr[-1] for tr in self.tracks]:
            for x,y in [tr[-1] for tr in self.tracks]:
                cv2.circle(frame_blobs, (np.int32(x),np.int32(y)), 3,(0,0,255), -1)
                #cv2.circle(frame_blobs, (np.int32(x),np.int32(y)), int(math.sqrt(area/math.pi)),(0,0,255), -1)

            # Combine and display original image and blobs image
            window_image = np.hstack((frame, frame_blobs))
            cv2.imshow('frame',window_image)
            
            # Calculate framerate
            framerate = cv2.getTickFrequency() / (cv2.getTickCount()- start)
            self.framerate_total += framerate

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            
        fdata.close()

        # Write fps and number of blobs to info.txt 
        framerate_average = self.framerate_total/self.frame_idx

        print "FPS: " + str(framerate_average)
        with open('info.txt', 'w') as f:
            f.write("FPS: " + str(framerate_average) + "\n")
        with open('info.txt', 'a') as f:
            f.write("Blobs: " + str(len(self.tracks)) + "\n")

        fdata = open("data.txt", "r")
        #while hasn't finished reading from data.txt
##        for i in xrange(1, len(self.tracks)):
##            with open('Data/Pin_' + str(i) + '.txt', 'w') as f:
##                f.write("Pin number " + str(i) + " data : \n")
##                f.write("----------------------------------------------\n")
##                f.write("X position |\t Y position \t | \t Area \n")
##                f.write("----------------------------------------------\n\n")

        line_n = 0
        for line in fdata: 
            with open('Data/Pin_' + str(line_n + 1) + '.txt', 'a') as f:
                f.write(line)
                line_n +=1
                line_n = line_n % len(self.tracks)
                
            
                


def main():
    
    # Select image folder
    folder = "Images"

    # Take images from folder and process them
    Lk(folder).run()

    # When everything is done, destroy windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


