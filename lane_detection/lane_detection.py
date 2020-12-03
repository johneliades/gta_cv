import cv2
import sys
import math
import numpy as np
import statistics as st

w = 800
h = 600
threshold = 100
show_orig = True

def init_lane_detection(width=800, height=600, show_original=True):
	global w
	global h
	global thresh2
	global show_orig

	w = width
	h = height

	show_orig = show_original

def draw_lines(orig_img, proc_img):
	global threshold
	global test

	lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, np.array([]), 20, 10)

	counter = -1
	try:
		group_lines_pos = []
		group_lines_neg = []

		ys = []
		for line in lines:
			x1,y1,x2,y2 = line[0]

			if(math.sqrt((x1-x2)**2 + (y1-y2)**2)<190):
				continue

			ys += [y1, y2]

		min_y = min(ys)
		max_y = h

		for line in lines:
			x1,y1,x2,y2 = line[0]
		
			if(math.sqrt((x1-x2)**2 + (y1-y2)**2)<190):
				continue

			counter += 1

			m = (y2-y1)/(x2-x1)
			b = y1-m*x1

			try:
				x1 = int((min_y - b)/m)
			except:
				continue

			x2 = int((max_y - b)/m)

			if(m>0):
				if(len(group_lines_pos)==0):
					group_lines_pos.append([(m, b, x1, min_y, x2, max_y)])
				else:
					found = False
					for group in group_lines_pos:
						cur = group[0]
						
						m_cur = cur[0]
						b_cur = cur[1]

						if(1.2 * abs(m_cur) > abs(m) > 0.8 * abs(m_cur)):
							if(1.2 * abs(b_cur) > abs(b) > 0.8 * abs(b_cur)):
								group.append((m, b, x1, min_y, x2, max_y))
								found = True
								break
				
					if(not found):
						group_lines_pos.append([(m, b, x1, min_y, x2, max_y)])
			elif(m<0):
				if(len(group_lines_neg)==0):
					group_lines_neg.append([(m, b, x1, min_y, x2, max_y)])
				else:
					found = False
					for group in group_lines_neg:
						cur = group[0]
						
						m_cur = cur[0]
						b_cur = cur[1]

						if(1.2 * abs(m_cur) > abs(m) > 0.8 * abs(m_cur)):
							if(1.2 * abs(b_cur) > abs(b) > 0.8 * abs(b_cur)):
								group.append((m, b, x1, min_y, x2, max_y))
								found = True
								break
				
					if(not found):
						group_lines_neg.append([(m, b, x1, min_y, x2, max_y)])

		counter_pos = []
		for group in group_lines_pos:
			counter_pos.append(len(group))

		def average_lane(lane_data):
			x1s = []
			y1s = []
			x2s = []
			y2s = []
			for data in lane_data:
				m, b, x1, y1, x2, y2 = data

				x1s.append(x1)
				y1s.append(y1)
				x2s.append(x2)
				y2s.append(y2)

			return (int(st.mean(x1s)), int(st.mean(y1s)), int(st.mean(x2s)), int(st.mean(y2s)))

		try:
			maximum1 = max(counter_pos) 
			index1 = [i for i, x in enumerate(counter_pos) if x == maximum1]
			index1 = index1[0]
			
			x1, y1, x2, y2 = average_lane(group_lines_pos[index1])
			cv2.line(orig_img, (x1, y1), (x2, y2), (255, 255, 0), 5)

			counter_neg = []
			for group in group_lines_neg:
				counter_neg.append(len(group))

			maximum2 = max(counter_neg)
			index2 = [i for i, x in enumerate(counter_neg) if x == maximum2]
			index2 = index2[0]

			x1, y1, x2, y2 = average_lane(group_lines_neg[index2])
			cv2.line(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
		except:
			threshold -= 2

		if(counter<4 and counter != -1):
			threshold -= 10
		elif(counter>10):
			threshold += 10

	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		#print(exc_type, exc_tb.tb_lineno)

def roi(proc_img):
	vertices = [np.array([[0, h],
						[0, 1.2*h/2],
						[3*w/8, h/3],
						[5*w/8, h/3],
						[w, 1.2*h/2],
						[w, h],
						], np.int32)]

	mask = np.zeros_like(proc_img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(proc_img, mask)
	return masked

def lane_detection(orig_img):
	# convert to gray
	proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

	# edge detection
	proc_img = cv2.Canny(proc_img, threshold1 = threshold, threshold2 = threshold+100)

	# range of interest
	proc_img = roi(proc_img)

	# apply blur to fix aliasing
	proc_img = cv2.GaussianBlur(proc_img, (5,5), 0)

	if(show_orig):
		draw_lines(orig_img, proc_img)
		return orig_img
	else:
		draw_lines(proc_img, proc_img)
		return proc_img
