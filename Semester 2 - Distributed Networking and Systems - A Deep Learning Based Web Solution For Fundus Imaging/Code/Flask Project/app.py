from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, send_from_directory
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.utils import secure_filename
import simplejson as json
import cv2
import numpy as np
import datetime

from DeepLearningFundusForFree.main_test import main_test

# UPLOAD_FOLDER = r'C:\Users\Prem Prasad\Desktop\MAIA\Semester 2 (Cassino)\Regular Semester\Distributed Programming and Networking\Flask\Flask Project\static'
UPLOAD_FOLDER = os.path.join('static')
UPLOAD_FOLDER_PATIENTS = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
HOMEPAGE_FOLDER = os.path.join('templates', 'images')
 
app = Flask(__name__)

# Chaning the Flask Application Configurations:
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/dpn'
app.config['SECRET_KEY'] = "random string"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_PATIENTS'] = UPLOAD_FOLDER_PATIENTS

# creating an object to interact with Database using SQLAlchemy
db = SQLAlchemy(app)

class doctors(db.Model):
	id = db.Column('sr_no', db.Integer, primary_key = True, auto_increment = True)
	doc_id = db.Column('doc_id', db.Integer, primary_key = True)
	doc_name = db.Column('doc_name', db.String(20))
	doc_pw = db.Column('doc_pw', db.String(20))

	def __init__(self, doc_id, doc_name, doc_pw):
		self.doc_id = doc_id;
		self.doc_name = doc_name;
		self.doc_pw = doc_pw;

class patients(db.Model):
	id = db.Column('sr_no', db.Integer, primary_key = True, auto_increment = True)
	pat_id = db.Column('pat_id', db.Integer, primary_key = True)
	doc_id = db.Column('doc_id', db.Integer)
	pat_name = db.Column('pat_name', db.String(20))
	pat_age = db.Column('pat_age', db.Integer)
	pat_gender = db.Column('pat_gender', db.String(20))
	pat_dur = db.Column('pat_dur', db.Integer)
	pat_gluc = db.Column('pat_gluc', db.Integer)
	pat_bp = db.Column('pat_bp', db.Integer)

	def __init__(self, pat_id, doc_id, pat_name, pat_age, pat_gender, pat_dur, pat_gluc, pat_bp):
		self.pat_id = pat_id;
		self.doc_id = doc_id;
		self.pat_name = pat_name;
		self.pat_age = pat_age;
		self.pat_gender = pat_gender;
		self.pat_dur = pat_dur;
		self.pat_gluc = pat_gluc;
		self.pat_bp = pat_bp; 

class images(db.Model):
	id = db.Column('sr_no', db.Integer, primary_key = True, auto_increment = True)
	pat_id = db.Column('pat_id', db.Integer, primary_key = True)
	img_name = db.Column('img_name', db.String(20))
	comment = db.Column('comment', db.String(100))
	date = db.Column('date', db.String(20))

	def __init__(self, pat_id, img_name, comment, date):
		self.pat_id = pat_id;
		self.img_name = img_name;
		self.comment = comment;
		self.date = date


@app.route('/')
def home():	
	# return render_template('sign_in.html');
	if not session.get('logged_in'):
		ban3 =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban3.jpg')
		ban1 =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban1.jpg')
		ban2 =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban2.jpg')
		ban4 =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban4.jpg')
		banner2 =  os.path.join(app.config['UPLOAD_FOLDER'], 'banner2.png')
		logo =  os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
		team1 = os.path.join(app.config['UPLOAD_FOLDER'], 'team1.jpg')
		team2 = os.path.join(app.config['UPLOAD_FOLDER'], 'team2.jpeg')
		team3 = os.path.join(app.config['UPLOAD_FOLDER'], 'team3.jpg')

		return render_template('home_page.html', 
								ban1 = ban1, 
								ban2 = ban2, 
								ban4 = ban4, 
								logo = logo,
								banner2 = banner2,
								team1 = team1,
								team2 = team2,
								team3 = team3)
		# file_full_path =  os.path.join(app.config['UPLOAD_FOLDER'], 'eyes.png')
		# return render_template('login.html')
		# login_image = file_full_path
	else:
		return render_template('doctorHomePage.html')



@app.route('/return_login')
def return_login():
	return render_template('login.html')
 
@app.route('/login', methods=['POST'])
def do_admin_login():

	form_username = request.form['username']
	form_password = request.form['password']
	

	# print(form_username)
	# print(form_password)

	# check if user exists:
	try:
		stmt = text("SELECT * from doctors WHERE doc_name = '{}'".format(form_username))
		query_result = db.engine.execute(stmt)
		names = [row[0] for row in query_result]
		
		print(".")
		print(".")
		print("Checking if user is present in database: \n")
		print(".")
		print(".")
		
		if len(names) == 0:
			print("user not present\n")
			flash("User does not exist. Please register")
			return render_template('login.html')
		
		else:
			print(".")
			print(".")
			print("user already existing! successful login!")
			print(".")
			print(".")

			query_result = doctors.query.filter_by(doc_name = form_username)
			actual_password = query_result['doc_name' == form_username].doc_pw

			print('actual_password: ')
			print(actual_password)

			if form_password == actual_password: 
				print('logged in successful') 
				session['logged_in'] = True
			else:
				# print(1)
				flash('wrong password!')
			return home()			
				
	except:
		print('error')




	

@app.route('/register')
def register():
	return render_template('register.html')

@app.route('/trial', methods=['POST'])
def trail():
	stmt = text("SELECT pat_id from patients")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8") 
	print("Patient id:")
	id_t  = request.form['patient_id']
	pat_id = id_t
	print(request.form['patient_id'])
	print("date:")
	print(request.form['date'])
	dt_t = request.form['date']   
	#date_time_obj = datetime.datetime.strptime(str(dt_t), '%a, %d %b %Y %H:%M:%S GMT')
	stmt = text("SELECT img_name from images where pat_id =" + str(id_t) +" AND date =\'" + str(dt_t)+"\'")
	print(stmt)
	query_result = db.engine.execute(stmt)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	val =(result.response[0]).decode("utf-8") 
	print(val)
	resp_dict = json.loads(val)
	print(resp_dict)
	test = resp_dict['result']
	test = test[0]
	print(test["img_name"])
	filename = test["img_name"]

		
	stmt = text("SELECT comment from images where pat_id =" + str(id_t) +" AND date =\'" + str(dt_t)+"\'")
	print(stmt)
	query_result = db.engine.execute(stmt)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	val =(result.response[0]).decode("utf-8") 
	print(val)
	resp_dict = json.loads(val)
	print(resp_dict)
	comment = resp_dict['result']
	comment = comment[0]
	print(comment["comment"])
	
	print('Processing complete')
	orig_img =  os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'original', pat_id, filename)
	seg_img =  os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'segmented', pat_id, filename)

	return render_template('upload_segment.html', disp = 1, orig_img = orig_img, seg = 1,
			seg_img = seg_img, pat_id_hidden = pat_id, img_name_hidden = filename,value = ret_val,comment = comment, edit =1)	


@app.route('/new_user', methods=['POST'])
def new_user():
	print("route: new user creation")
	form_username = request.form['username']
	form_id = request.form['id']
	form_password = request.form['password']
	form_confirm_password = request.form['confirm_password']
	
	if form_username == "" or form_password == "" or form_confirm_password == "":
		return render_template('register.html')


	if form_password == form_confirm_password:
		print(".") 
		print(".")
		print("passwords matched!")
		print(".") 
		print(".")
		
		# check if user exists:
		try:
			stmt = text("SELECT * from doctors WHERE doc_name = '{}'".format(form_username))
			query_result = db.engine.execute(stmt)
			names = [row[0] for row in query_result]
			
			print("Checking if user already exists: \n")

			if len(names) == 0:
				print("not present\n")
				# new user
				user = doctors(form_id, form_username, form_password)
				db.session.add(user)
				db.session.commit()
				print('successfully added new user to database!')

				flash("New User Added Successfully!")
				flash("Please login with created password")
				return render_template('login.html')
			else:
				print(".")
				print(".")
				print("user already existing! redirecting to login page...")
				print(".")
				print(".")
				return render_template('login.html')
		
		except:
			print('error')
	else:
		print("passwords don't match!")
		flash("Passwords don't match!")	
		return render_template('login.html', error = error)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(orig, seg):
	print('function called...')
	main_test(orig, seg)


@app.route("/segmentation", methods = ['GET', 'POST'])
def segmentation():
	#check if post request has 'file' part

	print('POST request to upload file')
	if 'file' not in request.files:
		print('No file part')
		flash('No file part')
		return render_template('upload_segment.html', disp = 1,edit = 0)

	#if user does not select a file, browser sends an empty file
	# so check if file has a filename to verify file has been uploaded
	file = request.files['file']
	pat_id = request.form['pat_id']
	print("WTF")
	print(pat_id)

	if file.filename == '':
		print('No file uploaded')
		flash('No file uploaded')
		return render_template('upload_segment.html', disp = 1,edit = 0)

	if pat_id == '':
		print('No patientID provided')
		# flash('No file uploaded')
		return render_template('upload_segment.html', disp = 1,edit = 0)

	if file and allowed_file(file.filename):
		print('filename allowed...')
		filename = secure_filename(file.filename)

		# print(os.path.join(os.getcwd() ,'static', 'uploads'))
		all_dirs = os.listdir(os.path.join(os.getcwd() ,'static', 'uploads', 'original'))
		print('all dirs: ')
		print(all_dirs)

		if pat_id not in all_dirs:
			print('patient directory not present. Creating one..')
			os.mkdir(os.path.join(os.getcwd() ,'static', 'uploads', 'original', pat_id))
			os.mkdir(os.path.join(os.getcwd() ,'static', 'uploads', 'segmented', pat_id))

		print(file.filename)
		
		
		path_orig = os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'original', pat_id, filename)
		path_seg  = os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'segmented', pat_id)
		file.save(path_orig)
		print('aaaaaaaaaaa Paths:')
		print(path_orig)
		print(path_seg)
		
		
		processed_img = process_image(path_orig, path_seg )


		print('Processing complete')
		orig_img =  os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'original', pat_id, filename)
		seg_img =  os.path.join(app.config['UPLOAD_FOLDER_PATIENTS'], 'segmented', pat_id, filename)
		print('bbbbbbbbbbbbaaaaaaaaa path2')
		print(orig_img)
		print(seg_img)
		stmt = text("SELECT pat_id from patients")
		print(stmt)
		query_result = db.engine.execute(stmt)
		print(query_result)
		result= jsonify({'result': [dict(row) for row in query_result]})
		print("here")
		print(result)
		print(result.response)
		print(type(result.response))
		print(len(result.response))
		print("there")
		print(type((result.response[0]).decode("utf-8") ))
		print((result.response[0]).decode("utf-8"))
		ret_val =(result.response[0]).decode("utf-8") 
		return render_template('upload_segment.html', disp = 1, orig_img = orig_img, seg = 0,
				seg_img = seg_img, pat_id_hidden = pat_id, img_name_hidden = filename,edit = 0,value = ret_val)		

	# if file and allowed_file(file.filen-ame):
	# 	print('filename allowed...')
#           filename = secure_filename(file.filename)
#           file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#           print('file uploaded successfully...')
#           return render_template('upload_image.html')	



from flask import jsonify 
@app.route("/load_add_patient_s")
def load_add_patient_s():
	# print('load_add_patient called')
	return render_template('add_patient.html')

@app.route("/load_add_patient", methods=['POST'])
def load_add_patient():
	# print('load_add_patient called')
	return render_template('add_patient.html')

@app.route("/profile")
def profile():
	# print('load_add_patient called')
	print('profile patient called')
	stmt = text("SELECT pat_id from patients")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8") 
	return render_template('patient_profile.html',value = ret_val,drop = 0)

@app.route("/add_comments", methods=['POST'])
def add_comments():
	print("here")
	pat_id_hidden = request.form['pat_id_hidden']
	print(pat_id_hidden)
	pat_id_hidden = int(pat_id_hidden)
	
	img_name_hidden = request.form['img_name_hidden']
	print(img_name_hidden)

	comments = request.form['comments']
	print(comments)

	comment = images(pat_id_hidden, img_name_hidden, comments, str(datetime.datetime.now()).split('.')[0])
	print('insertion step 1')
	db.session.add(comment)
	print('insertion step 2')
	db.session.commit()
	print('successfully added new user to database!')

	return render_template('temp_main.html')

@app.route("/search_patient_s")
def search_patient_s():
	print('search patient called')
	stmt = text("SELECT pat_id,date from images")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8") 
	return render_template('search.html',value = ret_val)




@app.route("/search_patient_f", methods=['POST'])
def search_patient_f():
	pat_id_hidden = request.form['pat_id_hidden']
	print(pat_id_hidden)
	pat_id_hidden = int(pat_id_hidden)
	
	img_name_hidden = request.form['img_name_hidden']
	print(img_name_hidden)

	comments = request.form['comments']
	print(comments)
	if 'Send' in request.form:
		comment = images(pat_id_hidden, img_name_hidden, comments, str(datetime.datetime.now()).split('.')[0])
		print('insertion step 1')
		db.session.add(comment)
		print('insertion step 2')
		db.session.commit()
		print('successfully added new user to database!')
		print('search patient called')
		stmt = text("SELECT pat_id,date from images")
		print(stmt)
		query_result = db.engine.execute(stmt)
		print(query_result)
		result= jsonify({'result': [dict(row) for row in query_result]})
		print("here")
		print(result)
		print(result.response)
		print(type(result.response))
		print(len(result.response))
		print("there")
		print(type((result.response[0]).decode("utf-8") ))
		print((result.response[0]).decode("utf-8"))
		ret_val =(result.response[0]).decode("utf-8") 
		return render_template('search.html',value = ret_val)
	if 'Edit' in request.form:
		str_stuf = "DELETE from images WHERE pat_id = " + str(pat_id_hidden) + " AND img_name = \'" + img_name_hidden +"\'"
		stmt = text(str_stuf)
		print(stmt)
		query_result = db.engine.execute(stmt)
		comment = images(pat_id_hidden, img_name_hidden, comments, str(datetime.datetime.now()).split('.')[0])
		print('insertion step 1')
		db.session.add(comment)
		print('insertion step 2')
		db.session.commit()
		print('successfully added new user to database!')
		print('search patient called')
		stmt = text("SELECT pat_id,date from images")
		print(stmt)
		query_result = db.engine.execute(stmt)
		print(query_result)
		result= jsonify({'result': [dict(row) for row in query_result]})
		print("here")
		print(result)
		print(result.response)
		print(type(result.response))
		print(len(result.response))
		print("there")
		print(type((result.response[0]).decode("utf-8") ))
		print((result.response[0]).decode("utf-8"))
		ret_val =(result.response[0]).decode("utf-8") 
		return render_template('search.html',value = ret_val)

@app.route("/add_patient", methods = ['POST'])
def add_patient():
	print('function call for adding patient...')
	form_id = request.form['id']
	form_doc_id = request.form['doc_id']
	form_name = request.form['name']
	form_age = request.form['age']
	form_gender = request.form['gender']
	form_duration = request.form['duration']
	form_glucose = request.form['glucose']
	form_bp = request.form['bp']

	# check if user exists:
	try:
		stmt = text("SELECT * from patients WHERE pat_id = '{}'".format(form_id))
		query_result = db.engine.execute(stmt)
		names = [row[0] for row in query_result]
		
		print("Checking if patient already exists: \n")

		if len(names) == 0:
			print("not present\n")
			# new user
			patient = patients(form_id, form_doc_id, form_name, form_age, form_gender, form_duration, form_glucose, form_bp)
			print('yes.1')
			db.session.add(patient)
			print('yes.2')
			db.session.commit()
			print('successfully added new patient to database!')

			return render_template('add_patient.html')
		else:
			print(".")
			print(".")
			print("user already existing! redirecting to login page...")
			print(".")
			print(".")
			return render_template('add_patient.html')
	
	except:
		print('error')


@app.route("/edit_patient", methods = ['POST'])
def edit_patient():
	print('function call for editing patient...')
	form_id = request.form['id']
	print("HELLO")
	form_doc_id = request.form['doc_id']
	form_name = request.form['name']
	form_age = request.form['age']
	form_gender = request.form['gender']
	form_duration = request.form['duration']
	form_glucose = request.form['glucose']
	form_bp = request.form['bp']
	print("HELLO1")
	# check if user exists:
	try:
		stmt = text("SELECT * from patients WHERE pat_id = '{}'".format(form_id))
		query_result = db.engine.execute(stmt)
		names = [row[0] for row in query_result]
		
		print("Checking if patient already exists: \n")

		if len(names) != 0:
			str_stuf = "DELETE from patients WHERE pat_id = " + str(form_id)
			stmt = text(str_stuf)
			print(stmt)
			query_result = db.engine.execute(stmt)
			patient = patients(form_id, form_doc_id, form_name, form_age, form_gender, form_duration, form_glucose, form_bp)
			print('yes.1')
			db.session.add(patient)
			print('yes.2')
			db.session.commit()
			return render_template('add_patient.html')
		else:
			print(".")
			print(".")
			print("user already existing! redirecting to login page...")
			print(".")
			print(".")
			return render_template('add_patient.html')
	
	except:
		print('error')


@app.route("/load_upload_segment", methods = ['POST'])
def load_upload_segment():

	logo =  os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
	# background =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban3.jpg')

	# return render_template('upload_new.html', logo = logo)
	print('segmentation patient called')
	stmt = text("SELECT pat_id from patients")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8") 
	return render_template('upload_segment.html', disp = 1, seg = 1,value = ret_val,edit = 0)

@app.route("/load_upload_segment_s")
def load_upload_segment_s():

	logo =  os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
	# background =  os.path.join(app.config['UPLOAD_FOLDER'], 'ban3.jpg')

	# return render_template('upload_new.html', logo = logo)
	print('segmentation patient called')
	stmt = text("SELECT pat_id from patients")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8") 
	return render_template('upload_segment.html', disp = 0,seg = 1,value = ret_val,edit = 0)

@app.route("/populate", methods = ['POST'])
def populate():

	print('profile patient called')
	stmt = text("SELECT pat_id from patients")
	print(stmt)
	query_result = db.engine.execute(stmt)
	print(query_result)
	result= jsonify({'result': [dict(row) for row in query_result]})
	print("here")
	print(result)
	print(result.response)
	print(type(result.response))
	print(len(result.response))
	print("there")
	print(type((result.response[0]).decode("utf-8") ))
	print((result.response[0]).decode("utf-8"))
	ret_val =(result.response[0]).decode("utf-8")
 
	pat = request.form['pat_id']
	print(pat)
	if (pat == "Select a Patient"):
		return render_template('patient_profile.html',value = ret_val, drop = 0) 
	print(pat)
	print("hello")
	if 'get_info' in request.form:
		str_stuf = "SELECT * from patients WHERE pat_id = " + str(pat)
		stmt = text(str_stuf)
		print(stmt)
		query_result = db.engine.execute(stmt)
		print(query_result)
		result= jsonify({'result': [dict(row) for row in query_result]})
		print("here")
		print(result)
		print(result.response)
		print(type(result.response))
		print(len(result.response))
		print("there")
		print(type((result.response[0]).decode("utf-8") ))
		print((result.response[0]).decode("utf-8"))
		ret_val2 =(result.response[0]).decode("utf-8") 
		# print('load_add_patient called')
		return render_template('patient_profile.html',patient_id = pat, value = ret_val,value2 = ret_val2, drop = 1)	
	if 'edit_info' in request.form:		
		str_stuf = "SELECT * from patients WHERE pat_id = " + str(pat)
		stmt = text(str_stuf)
		print(stmt)
		query_result = db.engine.execute(stmt)
		print(query_result)
		result= jsonify({'result': [dict(row) for row in query_result]})
		print("here")
		print(result)
		print(result.response)
		print(type(result.response))
		print(len(result.response))
		print("there")
		print(type((result.response[0]).decode("utf-8") ))
		print((result.response[0]).decode("utf-8"))
		ret_val2 =(result.response[0]).decode("utf-8") 
		# print('load_add_patient called')
		return render_template('edit_patient.html',patient_id = pat,value = ret_val,value2 = ret_val2)


@app.route("/logout")
def logout():
	session['logged_in'] = False
	return home()
 

if __name__ == "__main__":
	app.secret_key = os.urandom(12)
	app.run(debug=True,host='0.0.0.0', port=4000)
