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

old_pos = []
old_neg = []
def draw_lines(orig_img, proc_img):
	global threshold
	global test

	lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, np.array([]), 10, 10)

	try:
		group_lines_pos = []
		group_lines_neg = []

		ys = []
	
		try:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				if(math.sqrt((x1-x2)**2 + (y1-y2)**2)<150):
					continue
				ys += [y1, y2]
		except:
			threshold -= 5
			return

		try:
			min_y = min(ys)
		except:
			min_y = 0

		max_y = h

		counter = 0
		for line in lines:
			x1,y1,x2,y2 = line[0]

			counter += 1

			if(math.sqrt((x1-x2)**2 + (y1-y2)**2)<150):
				continue

			counter -= 1

			m = (y2-y1)/(x2-x1)
			b = y1-m*x1

			try:
				x1 = int((min_y - b)/m)
			except:
				continue

			x2 = int((max_y - b)/m)

			similarity = 0.15
			
			if(m>0):
				if(len(group_lines_pos)==0):
					group_lines_pos.append([(m, b, x1, min_y, x2, max_y)])
				else:
					found = False
					for group in group_lines_pos:
						cur = group[0]
						
						m_cur = cur[0]
						b_cur = cur[1]

						if((1 + similarity) * abs(m_cur) > abs(m) > (1 - similarity) * abs(m_cur)):
							if((1 + similarity) * abs(b_cur) > abs(b) > (1 - similarity) * abs(b_cur)):
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

						if((1 + similarity) * abs(m_cur) > abs(m) > (1 - similarity) * abs(m_cur)):
							if((1 + similarity) * abs(b_cur) > abs(b) > (1 - similarity) * abs(b_cur)):
								group.append((m, b, x1, min_y, x2, max_y))
								found = True
								break
				
					if(not found):
						group_lines_neg.append([(m, b, x1, min_y, x2, max_y)])

		def average_lane(lane_data, sign):
			global old_pos
			global old_neg

			cache_size = 5

			if(sign>0):
				old_pos.insert(0, lane_data)
				if(len(old_pos)>cache_size):
					old_pos.pop()
				current = old_pos
			else:
				old_neg.insert(0, lane_data)
				if(len(old_neg)>cache_size):
					old_neg.pop()
				current = old_neg

			x1s = []
			y1s = []
			x2s = []
			y2s = []

			for lane_data in current:
				for data in lane_data:
					m, b, x1, y1, x2, y2 = data

					x1s.append(x1)
					y1s.append(y1)
					x2s.append(x2)
					y2s.append(y2)

			return (int(st.mean(x1s)), int(st.mean(y1s)), int(st.mean(x2s)), int(st.mean(y2s)))

		try:
			counter_pos = []
			for group in group_lines_pos:
				counter_pos.append(len(group))

			maximum1 = max(counter_pos) 
			index1 = [i for i, x in enumerate(counter_pos) if x == maximum1]
			index1 = index1[0]
			x1, y1, x2, y2 = average_lane(group_lines_pos[index1], 1)
			cv2.line(orig_img, (x1, y1), (x2, y2), (255, 255, 0), 5)
		except:
			pass

		try:
			counter_neg = []
			for group in group_lines_neg:
				counter_neg.append(len(group))

			maximum2 = max(counter_neg)
			index2 = [i for i, x in enumerate(counter_neg) if x == maximum2]
			index2 = index2[0]
			x1, y1, x2, y2 = average_lane(group_lines_neg[index2], -1)
			cv2.line(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
		except:
			pass

		#print(counter)

		if(counter<4):
			threshold -= 15
		elif(counter>50):
			threshold += 15

	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print(exc_type, exc_tb.tb_lineno)

def roi(proc_img):
	vertices = [np.array([[0, h],
						[0, 1.2*h/2],
						[3*w/8, h/2.5],
						[5*w/8, h/2.5],
						[w, 1.2*h/2],
						[w, h],
						], np.int32)]

	mask = np.zeros_like(proc_img)
	cv2.fillPoly(mask, vertices, 255)

	masked = cv2.bitwise_and(proc_img, mask)

	vertices = [np.array([[0, h],
						[h/3.75, h],
						[h/3.75, h-h/3.75],
						[0, h-h/3.75],
						], np.int32)]

	mask = np.zeros_like(proc_img)
	cv2.fillPoly(mask, vertices, 255)
	mask = cv2.bitwise_not(mask)

	masked = cv2.bitwise_and(masked, mask)

	vertices = [np.array([[3.3*w/8, h/2],
						[3.3*w/8, h],
						[4.7*w/8, h],
						[4.7*w/8, h/2],
						], np.int32)]

	mask = np.zeros_like(proc_img)
	cv2.fillPoly(mask, vertices, 255)
	mask = cv2.bitwise_not(mask)

	masked = cv2.bitwise_and(masked, mask)

	return masked

def lane_detection(orig_img):
	# convert to gray
	proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

	# edge detection
	proc_img = cv2.Canny(proc_img, threshold1 = threshold, threshold2 = 2*threshold)

	# range of interest
	proc_img = roi(proc_img)

	#dilated_image = cv2.dilate(proc_img, np.ones((5,5), np.uint8))
	#proc_img = cv2.absdiff(dilated_image, proc_img)

	# apply blur to fix aliasing
	proc_img = cv2.GaussianBlur(proc_img, (5,5), 0)
	proc_img = cv2.normalize(proc_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

	if(show_orig):
		draw_lines(orig_img, proc_img)
		return orig_img
	else:
		draw_lines(proc_img, proc_img)
	
	return proc_img
