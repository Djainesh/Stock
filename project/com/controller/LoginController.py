from flask import request, render_template, redirect, url_for
from project import app
from project.com.vo.LoginVO import LoginVO
from project.com.dao.LoginDAO import LoginDAO
from flask import session

@app.route('/', methods=['GET'])
def adminLoadLogin():
    try:
        print("in login")
        return render_template('admin/login.html')
    except Exception as ex:
        print(ex)

@app.route("/admin/validateLogin", methods=['POST'])
def adminValidateLogin():
    try:
        loginUsername = request.form['loginUsername']
        loginPassword = request.form['loginPassword']

        loginVO = LoginVO()
        loginDAO = LoginDAO()

        loginVO.loginUsername = loginUsername
        loginVO.loginPassword = loginPassword
        loginVO.loginStatus = "active"

        loginVOList = loginDAO.validateLogin(loginVO)
        loginDictList = [i.as_dict() for i in loginVOList]
        print(loginDictList)
        lenLoginDictList = len(loginDictList)

        if lenLoginDictList == 0:
            msg = 'Username Or Password is Incorrect !'
            return render_template('admin/login.html', error=msg)
        elif loginDictList[0]['loginStatus'] == 'inactive':
            msg = 'Sorry,temporary you are blocked by Admin'
            return render_template('admin/login.html',error=msg)
        else:
            for row1 in loginDictList:
                loginId = row1['loginId']
                loginUsername = row1['loginUsername']
                loginRole = row1['loginRole']

                session['session_loginId'] = loginId
                session['session_loginUsername'] = loginUsername
                session['session_loginRole'] = loginRole
                session.permanent = True

                if loginRole == 'admin':
                    return redirect(url_for('adminLoadDashboard'))
                elif loginRole == 'user':
                    return redirect(url_for('userLoadDashboard'))
    except Exception as ex:
        print(ex)


@app.route('/admin/loadDashboard', methods=['GET'])
def adminLoadDashboard():
    try:
        print("in /admin/loadDashboard")
        if adminLoginSession() == 'admin':
            return render_template('admin/index.html')
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)


@app.route('/user/loadDashboard')
def userLoadDashboard():
    try:
        if adminLoginSession() == 'user':
            return render_template('user/index.html')
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)

@app.route('/admin/loginSession')
def adminLoginSession():
    try:
        print("in /admin/loginSession")
        print("<<<<<<<<<<<<<<<<True>>>>>>>>>>>>>>>>>>>>")
        if 'session_loginId' and 'session_loginRole' in session:
            if session['session_loginRole'] == 'admin':
                return 'admin'
            elif session['session_loginRole'] == 'user':
                return 'user'
            print("<<<<<<<<<<<<<<<<True>>>>>>>>>>>>>>>>>>>>")
        else:
            print("<<<<<<<<<<<<<<<<False>>>>>>>>>>>>>>>>>>>>")
            return False
    except Exception as ex:
        print(ex)


@app.route("/admin/logoutSession", methods=['GET'])
def adminLogoutSession():
    try:
        print("in /admin/logoutSession")
        session.clear()

        return redirect(url_for('adminLoadLogin'))
    except Exception as ex:
        print(ex)

