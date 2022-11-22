from werkzeug.utils import secure_filename
import os
from flask import request, render_template, redirect, url_for
from project import app
from project.com.vo.DatasetVO import DatasetVO
from project.com.dao.DatasetDAO import DatasetDAO
from datetime import datetime
from project.com.controller.LoginController import adminLoginSession, adminLogoutSession


UPLOAD_FOLDER = 'project/static/adminResources/dataset/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/admin/loadDataset')
def adminLoadDataset():
    try :
        if adminLoginSession() == 'admin':
            return render_template('admin/addDataset.html')
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)

@app.route('/admin/insertDataset',methods=['POST'])
def adminInsertDataset():
    try:
        if adminLoginSession() == 'admin':

            datasetVO = DatasetVO()
            datasetDAO = DatasetDAO()

            file = request.files['file']
            print(file)

            datasetFileName = secure_filename(file.filename)
            print(datasetFileName)

            datasetFilePath = os.path.join(app.config['UPLOAD_FOLDER'])
            print(datasetFilePath)

            now = datetime.now()
            datasetUploadDate = now.strftime("%d/%m/%Y")
            print("date",datasetUploadDate)
            datasetUploadTime = now.strftime("%H:%M:%S")
            print("time",datasetUploadTime)

            datasetVO.datasetUploadDate = datasetUploadDate
            datasetVO.datasetUploadTime = datasetUploadTime

            file.save(os.path.join(datasetFilePath, datasetFileName))
            datasetVO.datasetFileName = datasetFileName
            datasetVO.datasetFilePath = datasetFilePath.replace("project", "..")
            datasetDAO.insertDataset(datasetVO)

            return redirect(url_for('adminViewDataset'))
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)

@app.route('/admin/viewDataset')
def adminViewDataset():
    try:
        if adminLoginSession() == 'admin':
            datasetDAO = DatasetDAO()
            datasetVOList = datasetDAO.viewDataset()
            print("__________________", datasetVOList)
            return render_template('admin/viewDataset.html', datasetVOList=datasetVOList)
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteDataset')
def adminDeleteDataset():
    try:
        if adminLoginSession() == 'admin':
            datasetVO = DatasetVO()
            datasetDAO = DatasetDAO()
            datasetId = request.args.get('datasetId')
            datasetVO.datasetId = datasetId
            datasetDAO.deleteDataset(datasetVO)
            return redirect(url_for('adminViewDataset'))
        else:
            return adminLogoutSession()
    except Exception as ex:
        print(ex)

