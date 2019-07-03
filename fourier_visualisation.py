import cv2
import numpy as np
import cmath
import math
import time
from svg.path import parse_path

img_size_w = 1280		
img_size_h = 720

complex_plane_scale = 0.8
complex_plane_center = (int(img_size_w/2), int(img_size_h/2))

def wave(c, f, t):
	return c * cmath.exp(complex(0,  -f *  2 * math.pi* t))

def drawWave(image, center, arrow ):
	assert(len(center) == 2)
	length = (int)(complex_plane_scale* np.linalg.norm(np.array(arrow)))
	p = (center[0] + complex_plane_scale * arrow.real, center[1] + complex_plane_scale * arrow.imag)
	p_int = tuple(np.asarray(p).astype(int))
	center = tuple(np.asarray(center).astype(int))
	cv2.circle(image, center, length, (0,255,0), 1)
	cv2.arrowedLine(image, center, p_int, (0,0,255), 2)
	return p

def drawFourierFunction(image, center,  coeff_list, t):
	p = center
	for i in range(len(coeff_list)):
		p =  drawWave(image, p, wave(coeff_list[i][0], coeff_list[i][1], t))
	return p

def drawLines(image, offset, points, color = (255,0, 0), line_width = 1):
	for i in range(1,len(points)):
		p1 = tuple((np.asarray(points[i-1]).astype(int) + offset))
		p2 = tuple((np.asarray(points[i]).astype(int) + offset))
		cv2.line(image, p1, p2, color, line_width)

def get_function_points(svg_file, sample_path_nb = 1000):
	from xml.dom import minidom

	doc = minidom.parse(svg_file)  # parseString also exists
	paths = [parse_path(path.getAttribute('d')) for path
	                in doc.getElementsByTagName('path')]
	doc.unlink()
	points=[]
	path = max(paths, key=lambda p: len(p))

	for i in range(sample_path_nb):
		p = path.point(i/float(sample_path_nb))
		points.append(p) 
	return np.array(points)

def computeFourier(points, f):
	dt = 1.0/points.shape[0]
	t = np.arange(points.shape[0]) * dt
	c = np.exp(2j * math.pi * -f *t)
	prod = np.multiply(c,points) * dt 
	return np.sum(prod)

def draw_svg_points(image, svg_file, sample_path_nb = 1000):
		points = get_function_points(svg_file, sample_path_nb)
		img_points = [(int(p.real), int(p.imag)) for p in points]
		drawLines(image, (0,0), img_points)

def computeCoef(points, n):
	coef =[]
	for i in range(1,n):
		c = computeFourier(points, i)
		coef.append((c, i))
		c = computeFourier(points, -i)
		coef.append((c, -i))
	return coef



function_points = get_function_points("dino.svg", 500)

start_seconds = time.time()
max_points = 1000
image = np.zeros((img_size_h,img_size_w,3), np.uint8)

n = 40
seconds_per_cycle = 10
points = []
coef = computeCoef(function_points, n)

def update_scale(val):
	global complex_plane_scale,points,start_seconds
	complex_plane_scale = (val+1)/10
	points = []
	start_seconds = time.time()


def update_nb_waves(val):
	global points, coef ,start_seconds
	points = []
	coef = computeCoef(function_points, int(val+1))
	start_seconds = time.time()

def update_time_interval(val):
	global points, start_seconds, seconds_per_cycle
	start_seconds = time.time()
	points = []
	seconds_per_cycle = val+1

cv2.namedWindow('FourierVisualiser')
cv2.createTrackbar('Waves Nb','FourierVisualiser',n,255,update_nb_waves)
cv2.createTrackbar('Scale','FourierVisualiser',int(complex_plane_scale * 10),100,update_scale)
cv2.createTrackbar('Sec/Cycle','FourierVisualiser',seconds_per_cycle,30,update_time_interval)

while True:
	image = np.zeros((img_size_h,img_size_w,3), np.uint8)
	t = time.time() - start_seconds
	t = t/seconds_per_cycle
	p = drawFourierFunction(image, complex_plane_center, coef, t )
	if len(points) < max_points:
		points.append(p)

	drawLines(image, (0,0), points,(255,0, 255), 2)
	cv2.imshow('FourierVisualiser',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()