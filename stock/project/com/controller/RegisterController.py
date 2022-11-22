import random
import smtplib
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#from flask import render_template,request
from flask import render_template, url_for, redirect, request
from project import app
from project.com.dao.LoginDAO import LoginDAO
from project.com.dao.RegisterDAO import RegisterDAO
from project.com.vo.LoginVO import LoginVO
from project.com.vo.RegisterVO import RegisterVO
from project.com.controller.LoginController import adminLoginSession,adminLogoutSession

@app.route("/user/loadRegister")
def userLoadRegister():
    try:
        return render_template("user/register.html")
    except Exception as ex:
        print(ex)

@app.route("/user/insertRegister", methods=['POST'])
def userInsertRegister():
    try:
        loginUsername = request.form['loginUsername']
        loginPassword = ''.join((random.choice(string.ascii_letters + string.digits)) for x in range(8))
        print("password=", loginPassword)
        print("in insertRegister")

        loginVO = LoginVO()
        loginDAO = LoginDAO()

        sender = "shahmodipanchal@gmail.com"
        receiver = loginUsername
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['subject'] = "ACCOUNT PASSWORD"
        msg.attach(MIMEText('Welcome to our system:'))
        msg.attach(MIMEText(loginPassword, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, "hithere12345")
        text = msg.as_string()
        server.sendmail(sender, receiver, text)

        loginVO.loginUsername = loginUsername
        loginVO.loginPassword = loginPassword
        loginVO.loginRole = "user"
        loginVO.loginStatus = "active"

        loginDAO.insertLogin(loginVO)
        print("loginId",loginVO.loginId)

        firstName = request.form['firstName']
        lastName = request.form['lastName']
        registerContactNo = request.form['registerContactNo']
        registerAddress = request.form['registerAddress']

        registerVO = RegisterVO()
        registerDAO = RegisterDAO()

        registerVO.firstName = firstName
        registerVO.lastName = lastName
        registerVO.registerContactNo = registerContactNo
        registerVO.registerAddress = registerAddress
        registerVO.register_LoginId = loginVO.loginId

        registerDAO.insertRegister(registerVO)
        server.quit()
        return render_template("admin/login.html")
    except Exception as ex:
        print(ex)

@app.route('/admin/viewUser')
def adminViewUser():
    try:
        if adminLoginSession() == 'admin':
            loginDAO = LoginDAO()
            loginVOList = loginDAO.viewUser()
            return render_template('admin/viewUser.html', loginVOList=loginVOList)
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)


@app.route('/admin/blockUser')
def adminBlockUser():
    try:
        if adminLoginSession() == 'admin':
            loginVO = LoginVO()
            loginDAO = LoginDAO()
            loginId = request.args.get('loginId')
            loginVO.loginId = loginId
            loginVO.loginStatus = 'inactive'
            loginDAO.blockUnblockUser(loginVO)
            #return redirect('/admin/viewUser')
            return redirect(url_for('adminViewUser'))
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)

@app.route('/admin/unblockUser')
def adminUnblockUser():
    try:
        if adminLoginSession() == 'admin':
            loginVO = LoginVO()
            loginDAO = LoginDAO()
            loginId = request.args.get('loginId')
            loginVO.loginId = loginId
            loginVO.loginStatus = 'active'
            loginDAO.blockUnblockUser(loginVO)
            #return redirect('/admin/viewUser')
            return redirect(url_for('adminViewUser'))
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)
