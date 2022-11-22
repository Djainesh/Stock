from project import app
from flask import render_template
from project.com.controller.LoginController import adminLoginSession, adminLogoutSession

# Admin-side methods
# @app.route('/admin/viewUser')
# def adminViewUser():
#     if adminLoginSession() == 'admin':
#         return render_template('admin/viewUser.html')
#     else:
#         return adminLogoutSession()

@app.route('/admin/viewDetection')
def adminViewDetection():
    if adminLoginSession() == 'admin':
        return render_template('admin/viewDetection.html')

# User-side methods

@app.route('/user/loadCamera')
def userLoadCamera():
    return render_template('user/addCamera.html')

@app.route('/user/viewCamera')
def userViewCamera():
    return render_template('user/viewCamera.html')

@app.route('/user/viewDetection')
def userViewDetection():
    return render_template('user/viewDetection.html')
