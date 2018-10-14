# This script is a GUI for labeling. 
# includes:
# 	1.  instructions for user (2 screens)
#	2.  labeling mode, where the user labels each graph and
#	    the results are written to a .csv file to be used for learning 
#		and recommendation 
#	3.	exit screen when labeling experiment is done
#
# constants, system_options, functions and imported files are listed by alphabetical order.

"""----------------------
	 Imported Files
----------------------"""
import matplotlib
matplotlib.use('TkAgg')

import csv
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation
import sys
import time
import tkinter as tk


from matplotlib.widgets import Button as matButton
from PIL import Image as pilim, ImageTk as pilimgtk
from pylab import *
from tkinter import *
from tkinter import messagebox


"""----------------------
		CONSTANTS
----------------------"""
INSTR_IDX = 1 # an indicator for number of "next" clicks
POINT_CONST = 0.46  # const to show the edges points clearly



"""----------------------
	  System Options
----------------------"""

# these two will work only if TESTING_MODE = False, 
# and only one of them has to be True. These parameters
# determine if command line params are files or directories 
DIR_OPTION = True 
FILES_OPTION = False


INSTRUCTIONS_ON = True # instructions for user are visible or not
NUM_TRIALS = 3 # num of manual files set for TESTING_MODE
PRINT_LOG = False  # when "True", debug prints are on the way


COLLECT_DATA_MODE = False # when "True", collects a lot of data on the same graph
RECOMMENDATION_MODE = True # when "True", helps the user by suggesting optional points
TESTING_MODE = True  # when "True", manual files, otherwise from command line


"""----------------------
		Functions
----------------------"""


"""------------------------------------------------
this function finishes the experiment of labeling,
shows the last screen to user to notify him
and say thanks :)
...
@(): no parameters for this function
----------------------------------------------- """
def finish_experiment():
	root = Tk()
	root.withdraw()
	messagebox.showinfo("Finished", "Thank you for participating!")




"""------------------------------------------------
gets an x_cord and retrives its matching position
(index) in time_vals list
------------------------------------------------"""
def get_index_in_list(x_cord, granularity = 0.02):
	global time_vals
	if granularity == 0:
		print("why granularity zero ? :(")
		# error code for bad granularity
		return -1
	x_cord = segmentation.RoundValueToGranularity(x_cord)
	factor = 1/granularity
	return int(x_cord*factor)


"""-------------------------------------------------------------
this funcion gets a point and returns the suggested points
in relate to it, using SUGGESTION API
NOTE: currently it is for testing, we need to connect it
	  to Avishay's API
...
@csv_name: name of file current shown as a graph
@curr_x: the x_cord of the point we want its suggestions
-------------------------------------------------------------"""
def get_recommended_points(csv_name="",curr_x=20.22):
	if curr_x>20:
		return [0.24, 23.46, 45.88]
	return [10.4,20.00,35.24]



"""------------------------------------------------
 this function alternates between two screens
of user instructions. when the "next" button is 
being clicked for the secind time, the instructions
are gone and the graphs are shown!
...
@frame: the canvas we put the instructions on
@img: the second instruction image, replaces the
	  first instruction image after first click
	  of "next button"
@root: the root of the tkinter GUI
 -----------------------------------------------"""

def instruction_switch(frame,img,root,button):
	global INSTR_IDX
	if PRINT_LOG:
		print(INSTR_IDX)
	if INSTR_IDX == 1 and button == 'next':
		frame.create_image(0,0,image=img,anchor=NW)
		root.title("Instructions- page 2/2")
		INSTR_IDX += 1
	elif INSTR_IDX == 2 and button == 'prev':
		frame.create_image(0,0,image=img,anchor=NW)
		root.title("Instructions- page 1/2")
		INSTR_IDX -= 1
	elif INSTR_IDX == 1 and button == 'prev':
		return
	else:
		root.destroy()


"""--------------------------------------------------
this function creates buttons 'next','reset','hint'
in GUI, after user clears the current choice (after
plt.clear() is called)
...
no params for this function
--------------------------------------------------"""
def make_buttons():
	global im_hint; global im_hint_off_1; global show_suggestions_axes
	next_button_axes = plt.axes([0.91,0.46,0.08,0.08])
	im_next = pilim.open("imgs/Toggle_next.png")
	next_graph_but = matButton(next_button_axes, '', color = 'wheat', hovercolor = 'oldlace', image = im_next)
	next_graph_but.on_clicked(show_next_graph)

	reset_button_axes = plt.axes([0.894,0.11,0.05,0.05])
	im_reset = pilim.open("imgs/reset.png")
	reset_choice_but = matButton(reset_button_axes, '', color = 'powderblue', image = im_reset)
	reset_choice_but.on_clicked(reset_selection)

	show_suggestions_axes = plt.axes([0.932,0.11,0.05,0.05])
	show_suggestions_but = matButton(show_suggestions_axes, '', color = 'white', image = im_hint_off_1)
	show_suggestions_but.on_clicked(redraw_graph)


"""------------------------------------------------
this function notifies the user that none of the
previous users has ever chosen the current point
selected
...
@(): no parameters for this function
----------------------------------------------- """
def notify_no_suggestions():
	root1 = Tk()
	root1.withdraw()
	messagebox.showinfo("the last point has never been chosen before","we have no suggestions for you... but hey!\n it means that you are creative!")
	root1.destroy()


"""----------------------------------------------------
this function handles sorting the points,
organizing them in a well-formed csv-format string
...
@file_key: name of current graph's file
@first_point: first point to be shown on graph
@graph_points_str: string to hold the current labeling
@last_point: last point to be shown on graph
----------------------------------------------------"""
def organize_graph_output(file_key,first_point,graph_points_str,last_point):
	#	after collecting data, prepare it to be written in the file:
	#	put the last point in it, and then newline
	graph_points_str =  first_point+","+graph_points_str
	graph_points_str += last_point

	graph_ordered_points = list(map(lambda x: float(x),graph_points_str.split(",")))
	graph_ordered_points.sort()

	# print(graph_ordered_points)
	graph_status_str = ''
	for point in graph_ordered_points:
		graph_status_str += str(point)+","

	return file_key+graph_status_str+"\n"


"""-----------------------------------------------------------------
this function creates a handler for mouse-pressing
event. it is meant for collecting the coordinates of the 
pressed location and pass them to the saved-points file.
NOTE: right now, it collects only time coords, if otherwise desired:
	  change manually
NOTE 2: right now, X_cord first choice of user is set to 0,
		last choice is set to @max_time
...
@event: the event causing the handler to work
		here specifically: 'left mouse button click'
-----------------------------------------------------------------"""
def press_and_get_cords(event):

	global graph_points_str
	global point_count
	global selected_points
	global time_vals
	global my_plot
	global show_suggestions_axes; global im_hint_off_1
	if event.inaxes != my_plot.axes:
		return
	if event.xdata != None and event.xdata > time_vals[-1]:
		return
	tb = get_current_fig_manager().toolbar
	in_pic = event.xdata != None and event.ydata != None #	validates that clicking is in pic
	if tb.mode == "" and in_pic:
		(x,y) = (event.xdata,event.ydata)
		point_count += 1
		point_str = str(x)+","
		graph_points_str+=point_str
		selected_points.append((x,y))
		#	for debug
		print('user pressed: ' + point_str)
		show_suggestions_axes.images[0].set_data(im_hint_off_1)
		show_selected_point(x)


"""-----------------------------------------------------
this function gets a .CSV file and shows its content as
as graph
...
@csv_name: name of input file
@fig_idx: listed-index of current graph
-----------------------------------------------------"""
def print_csv_as_graph(csv_name,fig_idx):
	global time_vals; global blue_vals; global red_vals
	global max_time; global max_blue; global max_red; global min_blue; global min_red
	global first_046_point; global last_002_point
	global my_plot
	global im_hint; global im_hint_off_1; global show_suggestions_axes
	
	with open(csv_name) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			time = float(row[0])
			blue = float(row[1])
			red = float(row[2])
			if time == 0.46:
				first_046_point = (red+blue)/2

			max_time,max_blue,max_red,min_blue,min_red = update_axes_vals(max_time, \
				max_blue,max_red,min_blue,min_red,time,blue,red)

			time_vals.append(time)
			blue_vals.append(blue)
			red_vals.append(red)

	last_002_point = (red_vals[get_index_in_list(max_time-0.02)]+\
				blue_vals[get_index_in_list(max_time-0.02)])/2


	"""debugging prints"""
	if PRINT_LOG:
		print("max time: ",math.ceil(max_time))
		print("max blue: ",max_blue)
		print("max red: ",max_red)


	my_plot = plt.subplot(111)
	fig = plt.gcf()
	fig.canvas.set_window_title("figure "+str(fig_idx))
	if PRINT_LOG:
		print("type of fig: ", type(fig)) #	type of fig:  <class 'matplotlib.figure.Figure'>


	reset_pic_params(time_vals,blue_vals,red_vals,max_time,max_blue,\
				max_red,min_blue,min_red,first_046_point,last_002_point)

	cid_press = fig.canvas.mpl_connect('button_press_event',press_and_get_cords)

	next_button_axes = plt.axes([0.91,0.46,0.08,0.08])
	im_next = pilim.open("imgs/Toggle_next.png")
	next_graph_but = matButton(next_button_axes, '', color = 'wheat', hovercolor = 'oldlace', image = im_next)
	next_graph_but.on_clicked(show_next_graph)

	reset_button_axes = plt.axes([0.894,0.11,0.05,0.05])
	im_reset = pilim.open("imgs/reset.png")
	reset_choice_but = matButton(reset_button_axes, '', color = 'powderblue', image = im_reset)
	reset_choice_but.on_clicked(reset_selection)

	show_suggestions_axes = plt.axes([0.932,0.11,0.05,0.05])
	show_suggestions_but = matButton(show_suggestions_axes, '', color = 'white', image = im_hint_off_1)
	show_suggestions_but.on_clicked(redraw_graph)
	plt.show()	

	return max_time



"""-------------------------------------------------------------
this funcion gets the current suggested points and re-draw
the graph with them.
...
@suggested_points: a list of the sugested points, represented
				   as tuples
-------------------------------------------------------------"""
def redraw_graph(event,mode = "redraw"):
	print("in redraw!\n\n")

	global graph_points_str; global point_count
	global time_vals; global blue_vals; global red_vals
	global max_time; global max_blue; global max_red; global min_blue; global min_red
	global first_046_point
	global selected_points; global suggested_points; global my_plot
	global im_hint; global show_suggestions_axes

	if selected_points == []:
		suggested_points = suggest_points(0) + suggest_points(time_vals[-1])
	print(suggested_points)

	if suggested_points == [] and mode == "redraw":
		# none of the user has ever chosen the current selected point,
		# so alert him
		notify_no_suggestions()

	my_plot.clear()
	reset_pic_params(time_vals,blue_vals,red_vals,max_time,max_blue,max_red,\
		min_blue,min_red,first_046_point,last_002_point)
	for point in selected_points:
		my_plot.plot(point[0], point[1], 'ko')
	if mode == "redraw":
		show_suggestions_axes.images[0].set_data(im_hint)
		for point in suggested_points:
			my_plot.plot(point[0], point[1], 'yo')
	plt.draw()



"""----------------------------------------------------------
set the parameters for plt. drawing, function of showing 
stuff on screen used by matplotlib.
...
@time_vals: list of time values, for x_axis
@blue_vals: list of blue values, for y_axis
@red_vals: list of red values, for y_axis
@max_time: max value of time in game
@max_blue: max velocity of blue in game
@max_red: max velocity of red in game
@min_blue: min velocity of blue in game
@min_red: min velocity of red in game
@first_046_point: first point to be shown on graph
@last_002_point: last point to be shown on graph
----------------------------------------------------------"""
def reset_pic_params(time_vals,blue_vals,red_vals,max_time, \
	max_blue,max_red,min_blue,min_red,first_046_point,last_002_point):
	
	global my_plot
	#	set pic axes and titles
	my_plot.set_xlabel('time (seconds)'); my_plot.set_ylabel('velocity (mm/sec)')
	time_bound, velocity_lower_bound, velocity_upper_bound \
		= set_axes_vals(max_time,max_blue,max_red,min_blue,min_red)
	my_plot.axis([0, time_bound, velocity_lower_bound, velocity_upper_bound])

	#	the graph itself
	my_plot.plot(time_vals,blue_vals,'b-', linewidth=0.75)
	my_plot.plot(time_vals,red_vals,'r-',linewidth=0.75)

	#	automatically shows the last point in cian and puts it into the labeling
	my_plot.plot(max_time-0.02, last_002_point, 'ko')
	my_plot.plot(POINT_CONST, first_046_point, 'ko')

	# make_buttons()

"""---------------------------------------------------------
 when left arrow is being hitten, the marks on
the current graph are all removed,
and the user's choice for the current graph is being reset
...
@event: the event causing the handler to work, 
		here specifically: 'left arrow hit'
---------------------------------------------------------"""
def reset_selection(event):
	if event.name != "button_release_event":
		return
	print("in reset ----",event.name)
	global graph_points_str; global point_count
	global time_vals; global blue_vals; global red_vals
	global max_time; global max_blue; global max_red; global min_blue; global min_red
	global first_046_point
	global selected_points
	global show_suggestions_axes; global im_hint_off_1

	my_plot.clear()
	reset_pic_params(time_vals,blue_vals,red_vals,max_time,max_blue,max_red,\
		min_blue,min_red,first_046_point,last_002_point)
	graph_points_str = ""
	point_count = 0
	selected_points=[]
	show_suggestions_axes.images[0].set_data(im_hint_off_1)
	plt.draw()



"""------------------------------------------------------------
this function gets the parameters for axes, manipulates
them to set the highest values of each axis: x, and y
and returns these values.
...
@time: time val to compute time bound
@max_blue: max blue velocity to compute upper velocity bound
@max_red: max red velocity to compute upper velocity bound
@min_blue: min blue velocity to compute lower velocity bound
@min_red: min red velocity to compute lower velocity bound
------------------------------------------------------------"""
def set_axes_vals(time,max_blue,max_red,min_blue, min_red):
	max_t = math.ceil(time) + 3
	min_v = min(min_blue,min_red) - 40
	max_v = max(max_blue,max_red) + 40
	return (max_t,min_v,max_v)


"""----------------------------------------------
 when right arrow is being hitten, the marks on
the current graph are written to the file,
and the next graph is shown
...
@event: the event causing the handler to work, 
		here specifically: 'right arrow hit'
----------------------------------------------"""
def show_next_graph(event):
	if event.name != "button_release_event":
		return
	plt.close()

"""-------------------------------------------------------------
 when the user clicks on a point in the graph, this 
point is marked in a black dot.
...
@point_x,point_y: @x and @y coordinates of the selected point
-------------------------------------------------------------"""
def show_selected_point(point_x=0):
	global my_plot
	redraw_graph(None,"regular")
	if RECOMMENDATION_MODE:
		suggest_points(point_x)
	plt.show()


"""------------------------------------------------
this function starts the experiment of labeling,
shows the instructions to user
...
@(): no parameters for this function
----------------------------------------------- """
def start_experiment():
	root = Tk()
	root.title("Instructions- page 1/2")
	instruction_1 = pilimgtk.PhotoImage(pilim.open("imgs/instr_1.png"))
	instruction_2 = pilimgtk.PhotoImage(pilim.open("imgs/instr_2.png"))
	next_but_pic = pilimgtk.PhotoImage(pilim.open("imgs/NEXT.png"))
	prev_but_pic = pilimgtk.PhotoImage(pilim.open("imgs/PREV.png"))
	frame = Canvas(root,bg="blue",width=625,height=460)
	frame.create_image(0,0,image=instruction_1,anchor=NW)
	frame.pack(expand=True)

	next_button = Button(root, text ="next",command=lambda: instruction_switch(frame,instruction_2,root,'next'),\
				 image = next_but_pic,padx = 5, pady = 4, font = ("Arial", 10), bd = 4)
	next_button.pack(side = RIGHT,fill = BOTH)

	prev_button = Button(root, text ="prev",command=lambda: instruction_switch(frame,instruction_1,root,'prev'),\
				 image = prev_but_pic,padx = 5, pady = 4, font = ("Arial", 10), bd = 4)
	prev_button.pack(side = RIGHT, fill = BOTH)

	root.mainloop()



"""-------------------------------------------------------------
this funcion gets a point in the graph, uses the SUGGESTION API
to get the suitable suggested points, puts them on the graph
and removes the previous suggestion.
...
@curr_x,curr_y: point in relevance to it we suggest other points
-------------------------------------------------------------"""
def suggest_points(curr_x):

	global csv_name
	global time_vals; global blue_vals; global red_vals
	global master; global suggested_points

	if PRINT_LOG:
		print("in suggest_points - file = ", csv_name)

	# gets the points to be suggested according to curr_x, get API from Avishay
	suggested_list_x = master.MakeSuggestions(csv_name, curr_x)

	# suggested_list now contain only x_values, we need to get their y_values.
	list_y = []
	list_x = []
	for x in suggested_list_x:
		if x != 0 and x != time_vals[-1]:
			x_index = get_index_in_list(x)
			print("blue size: ",len(blue_vals) ," and red size: ",len(red_vals))
			print("x: ",x," and index: ", x_index)
			list_x.append(x)
			list_y.append((blue_vals[x_index] + red_vals[x_index])/2)
	print("---------------------------------------------------")
	# calc suggested points vals
	suggested_points=[]
	for i in range(len(list_y)):
		suggested_points.append((list_x[i],list_y[i]))

	print("suggested: ",suggested_points)

	return suggested_points
	# send to re-draw
	# redraw_graph(suggested_points)


"""-------------------------------------------------
this function should update the high values of
the axes, computing through data reading.
...
@max_t: current max time val
@max_b: current max blue velocity val
@max_r: current max red velocity val
@min_b: current min blue velocity val
@min_r: current min red velocity val
@t,b,r: time,blue_v,red_v current read from file
-------------------------------------------------"""
def update_axes_vals(max_t,max_b,max_r,min_b,min_r,t,b,r):
	res = [max_t, max_b, max_r, min_b, min_r]
	if t >= max_t:
		res[0] = t
	if b >= max_b:
		res[1] = b
	if r >= max_r:
		res[2] = r
	if b <= min_b:
		res[3] = b
	if r <= min_r:
		res[4] = r
	return tuple(res)


"""---------------------------------------------------------
from now on - only code sections, no functions...
---------------------------------------------------------"""

"""----------------------------------------------"""
""" first two screens - instructions for the user"""
"""----------------------------------------------"""
if INSTRUCTIONS_ON:
	start_experiment()
	

"""-----------------------------------------------------
Two optional modes - one for windows, another for posix
(linux). Each mode has two options for the parmeters 
sent in the command line:
	(*) parameters are files: in this case, all files 
		are being run by the GUI
	(*) parameters are directories: in this case, all
		files in all directories are run bu GUI.

	*assumption: all files are .csv files !

	  
-----------------------------------------------------"""

filenames_list = []

"""determine the files for testing"""
if TESTING_MODE:
	for i in range(NUM_TRIALS):
		date = time.strftime("%d-%m-%y")
		hour = time.strftime("%H:%M:%S")
		labeling_fname = 'labeling_' + date + '_' + hour + '_' + str(i+1)
		filenames_list.append((i+1,'datasets/dataset-main/experts_[1]_trial_'+str(i+1)+'.csv','sessions/'+labeling_fname+'.csv'))
if COLLECT_DATA_MODE:
	for i in range(15):
		date = time.strftime("%d-%m-%y")
		hour = time.strftime("%H:%M:%S")
		labeling_fname = 'labeling_' + date + '_' + hour + '_' + str(i+1)
		filenames_list.append((i+1,'datasets/dataset-main/experts_[1]_trial_4.csv','sessions/'+labeling_fname+'.csv'))

else:
	if os.name == 'nt':
		if DIR_OPTION:
			dir_list = sys.argv[1:len(sys.argv)]
			i=0
			for dir in dir_list:
				print("dir: ",dir)
				path = os.listdir(".\\"+dir)
				for file in path:
					print("		file ",i,": ", file)
					filenames_list.append((i+1,file))
					i+=1
		elif FILES_OPTION:
			""" get the files from command line - as seperate files"""
			# files_list = sys.argv[1:len(sys.argv)]
			# i=0
			# for file in files_list:
			# 	print("		file ",i,": ", file)
			# 	filenames_list.append((i+1,file))
			# 	i+=1
	elif os.name == 'posix':
		pass
		# if DIR_OPTION:
		# 	dir_list = sys.argv[1:len(sys.argv)]
		# 	i=0
		# 	for dir in dir_list:
		# 		print("dir: ",dir)
		# 		path = os.listdir("./"+dir)
		# 		for file in path:
		# 			print("		file ",i,": ", file)
		# 			filenames_list.append((i+1,file))
		# 			i+=1


if PRINT_LOG:
	print(filenames_list)

graph_points_str=''
point_count = 0
time_vals = []; blue_vals = []; red_vals = []
max_time = 0; max_blue = 0; max_red = 0; min_blue = 0; min_red = 0
first_046_point = 0; last_002_point = 0
csv_name = ""
selected_points=[]; suggested_points = []
start_time = 0; end_time = 0
im_hint = pilim.open("imgs/hint_1.png")
im_hint_off_1 = pilim.open("imgs/hint_off_1.png")
show_suggestions_axes = None

# Summon the Learning Master!
if RECOMMENDATION_MODE:
	start_time = time.time()
	master = segmentation.Segmentation('datasets/dataset-main')
	master.Load('master-main')
	end_time = time.time()
	print("Learning time: "+str(end_time-start_time)+" seconds")


#	iterate all files and get them labeled by user:
#	filename[0] = serial num of figure
#	filename[1] = name of game .csv file
#	filename[2] = name of file storing data on previous labelings of filename[1]

for filename in filenames_list:
	labelingFile = open(filename[2],'w')
	file_key = str(filename[1].split(".")[0].split('/')[-1]) # simply takes the name of the game file, without '.csv' ending
	csv_name = file_key
	file_key +=","
	first_point = str(0)

	last_point = str(print_csv_as_graph(filename[1],filename[0]))

	graph_points_str = organize_graph_output(file_key,first_point,graph_points_str,last_point)

	labelingFile.write(graph_points_str)
	labelingFile.close()
	#	reset global variables for the next file :)
	graph_points_str=''; point_count = 0; time_vals=[]; blue_vals=[]; red_vals=[]
	max_time = 0; max_blue = 0; max_red = 0; min_blue = 0; min_red = 0
	selected_points = []

finish_experiment()
