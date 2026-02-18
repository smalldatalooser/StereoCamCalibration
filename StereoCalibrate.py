#!/usr/bin/env python
'''
StereoCalibrate is a simple application that uses OpenCv's stereo vision tools
to compute the calibration parameters required for further stereo processing of
images collected from a stereo camera system.

This application is a rough cut. It was originally written for internal AFSC use only
and has been working "well enough" so we haven't devoted much time to bringing
it to a release state. Take it as is. You're free to fix and/or extend as you see
fit. Please consider submitting any updates and bug fixes to us to improve the
application for all users.

rick.towler@noaa.gov
kresimir.williams@noaa.gov

'''
import sys,  os
import copy
import functools
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtSql
from ui import ui_StereoCalibrate
import imgfiledlg,  infodlg
import  string,  glob,  cv2
import numpy as np
import re
from math import  *

class StereoCalibrate(QMainWindow, ui_StereoCalibrate.Ui_StereoCalibrate):

    def __init__(self, resetWindowPosition=False,  parent=None):
        super(StereoCalibrate, self).__init__(parent)
        self.setupUi(self)

        self.pointMarks=[]
        self.cornerPointMarks=[]
        self.calPoints={}
        self.objPoints={}
        self.chessPoints={self.gvLeft:np.array([]),self.gvRight:np.array([])}
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2,2))
        self.kernel = np.ones((5,5), np.uint8)
        self.cornerPos={self.gvLeft: np.zeros([4, 2]), self.gvRight: np.zeros([4, 2])}
        self.cornerCount={self.gvLeft: 0, self.gvRight: 0}
        self.db=None
        self.workingPath=''
        self.imgnum=1

        self.crosshair_color = [240,10,240]
        self.marker_color = [10,10,240]
        self.text_color = [10,240,10]
        self.text_size = 12

        #  connect signals and slots
        
        
        
        self.loadBtn.clicked.connect(self.startNewCal)
        self.reloadCalBtn.clicked.connect(self.reloadCal)
        
        #self.connect(self.autoBtn, SIGNAL("clicked()"), self.autoFindCorners)
        #self.connect(self.getCornersBtn, SIGNAL("clicked()"), self.toggleCornerFinder)
        
        self.nextBtn.clicked.connect(self.advance)
        self.prevBtn.clicked.connect(self.advance)
        self.calBtn.clicked.connect(self.runCal)
        self.clearLeftBtn.clicked.connect(self.clearPoints)
        self.clearRightBtn.clicked.connect(self.clearPoints)
        self.scrubBox.stateChanged.connect(self.scrubImage)
        self.doneBtn.clicked.connect(self.saveAndExit)
        
        for obj in [self.gvLeft, self.gvRight]:
            #signals
            obj.mousePress.connect(self.imageClick)
            obj.setSelectionRadius(8)
            obj.autoWheelZoom=False
            obj.doSelections=False
            obj.selectAddKey=None
            obj.rubberBandKey=None

        #  restore the application state
        self.__appSettings = QSettings('afsc.noaa.gov', 'StereoCalibrate')
        size = self.__appSettings.value('winsize', QSize(940,646))
        position = self.__appSettings.value('winposition', QPoint(10,10))
        
        if not resetWindowPosition:
            #  make sure the program is on the screen
            position, size = self.checkWindowLocation(position, size)

            #  now move and resize the window
            self.move(position)
            self.resize(size)

        self.xDimEdit.setText(self.__appSettings.value('corner_finder_dim', '21'))
        self.wtEdit.setText(self.__appSettings.value('click_weight', '50'))
        self.harrisEdit.setText(self.__appSettings.value('harris_thresh', '0.2'))
        self.devThreshEdit.setText(self.__appSettings.value('dev_thresh', '200'))
        self.__dataDir= self.__appSettings.value('datadir', '')

        #  create an instance of the info dialog
        self.infodlg=infodlg.InfoDlg( resetWindowPosition, self)

        #  set the base directory path - this is the full path to this application
        self.baseDir = functools.reduce(lambda l,r: l + os.path.sep + r,
                os.path.dirname(os.path.realpath(__file__)).split(os.path.sep))
        try:
            self.setWindowIcon(QIcon(self.baseDir + os.sep + 'resources/checker.png'))
        except:
            pass


    def toggleCornerFinder(self):
        if self.getCornersBtn.isChecked():
            self.cornerPos[self.gvLeft]=np.zeros([4, 2])
            self.cornerPos[self.gvRight]=np.zeros([4, 2])
            self.cornerCount[self.gvLeft]=0
            self.cornerCount[self.gvRight]=0
        else:
            self.clearMarks()


    def checkWindowLocation(self, position, size):
        '''
        checkWindowLocation accepts a window position (QPoint) and size (QSize)
        and returns a potentially new position and size if the window is currently
        positioned off the screen.
        '''

        #  determine the current virtual screen size
        screenObj = QGuiApplication.primaryScreen()
        screenGeometry = screenObj.availableVirtualGeometry()

        #  assume the new and old positions are the same
        newPosition = position
        newSize = size

        #  check if the upper left corner of the window is off the screen
        if position.x() < screenGeometry.x():
            newPosition.setX(10)
        if position.x() >= screenGeometry.x():
            newPosition.setX(10)
        #  check if the window title bar is off the top of the screen
        if position.y() < screenGeometry.y():
            newPosition.setY(10)
        #  check if the window title bar is off the bottom of the screen
        #  Subtract 50 pixels from the position to ensure that
        #  the title bar is clear of the taskbar
        if (position.y() - 50) >= screenGeometry.y():
            newPosition.setY(10)

        #  now make sure the lower right (resize handle) is on the screen
        if (newPosition.x() + newSize.width()) > screenGeometry.width():
            newSize.setWidth(screenGeometry.width() - newPosition.x() - 5)
        if (newPosition.y() + newSize.height()) > screenGeometry.height():
            newSize.setWidth(screenGeometry.height() - newPosition.y() - 5)

        return [newPosition, newSize]


    def advance(self):

        #  check if we've started working with at least one of the images
        if self.dirty:

            #  check if we have detected corners on both images
            if (self.chessPoints[self.gvLeft].size > 0 and self.chessPoints[self.gvRight].size > 0):
                #  yes, we need to insert data into cal database - set the wait cursor
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

                #  get the image names
                pL = self.leftImages[self.shotList[self.num]].split(os.sep)
                pR = self.rightImages[self.shotList[self.num]].split(os.sep)
                imgLName = pL.pop()
                imgRName = pR.pop()

                #  delete the existing data for this frame pair
                QtSql.QSqlQuery("DELETE FROM points WHERE frame_number=" + str(self.shotList[self.num]))

                #  insert the new points
                ptNum = 1
                for i in range(self.chessPoints[self.gvLeft].shape[0]):
                    sql = ("INSERT INTO points (frame_number,left_image_filename,right_image_filename," +
                            "point_number,chess_pt_LX,chess_pt_LY,chess_pt_RX,chess_pt_RY,object_pt_X," +
                            "object_pt_Y,object_pt_Z) VALUES(" + str(self.shotList[self.num]) + ",'" + imgLName +
                            "','" +  imgRName+"'," + str(i) + "," + str(self.chessPoints[self.gvLeft][i, 0]) + "," +
                            str(self.chessPoints[self.gvLeft][i, 1]) + "," + str(self.chessPoints[self.gvRight][i, 0]) +
                            "," + str(self.chessPoints[self.gvRight][i, 1]) + "," + str(self.objectPoints[i, 0])+"," +
                            str(self.objectPoints[i, 1]) + "," + str(self.objectPoints[i, 2]) + ")")
                    QtSql.QSqlQuery(sql)
                    ptNum+=1

                #  clear the point arrays
                self.chessPoints[self.gvLeft] = np.array([])
                self.chessPoints[self.gvRight] = np.array([])

                #  and reset the wait cursor
                QApplication.restoreOverrideCursor()

            else:
                #  we've only worked on one image
                ok = QMessageBox.warning(self, "WARNING", "I don't have corners for both images, do you want " +
                        "to move on anyway? Any selected points will be discarded and this pair will " +
                        "not be used in the calibration.",  QMessageBox.Yes | QMessageBox.No)
                if (ok == QMessageBox.No):
                    return

        #  navigate forward or back
        if (self.sender() == self.nextBtn):
            self.num = min(self.num + 1, len(self.leftImages) - 1)
        elif (self.sender()==self.prevBtn):
            self.num = max(self.num - 1, 0)

        #  refresh the screen
        self.loadImagePair(self.shotList[self.num])


    def runCal(self):
        testing=False
        try:
            imsizeL=self.grayLeft.shape[::-1]
            imsizeR=self.grayRight.shape[::-1]
            devThresh=float(self.devThreshEdit.text())
            imInd=np.array(range(len(self.leftImages)))+1
            query=QtSql.QSqlQuery("SELECT frame_number, count(*) FROM points GROUP BY frame_number ORDER BY frame_number")
            procInd=[]
            numRec=[]
            while query.next():
                procInd.append(query.value(0))
                numRec.append(query.value(1))
            procInd=np.array(procInd)
            list=np.setdiff1d(imInd, procInd)
            if len(list)>0:
                numstr=''
                for num in list:
                    numstr=numstr+str(num)+' ,'
                reply=QMessageBox.warning(self, "WARNING", "Images "+numstr+" have not been processed. " +
                        "Is this ok?",  QMessageBox.Yes|QMessageBox.No)
                if reply==QMessageBox.No:
                    return

            objpoints=[]
            imgpointsL=[]
            imgpointsR=[]

            for i, num in enumerate(procInd):
                query=QtSql.QSqlQuery("SELECT chess_pt_LX, chess_pt_LY , chess_pt_RX, "+
                        "chess_pt_RY, object_pt_X, object_pt_Y,object_pt_Z FROM points WHERE frame_number="+str(num))
                c=0
                holderL=np.zeros((numRec[i], 1, 2))
                holderR=np.zeros((numRec[i], 1, 2))
                holderObj=np.zeros((numRec[i], 3))
                while query.next():
                    holderL[c, 0, :]=np.array([query.value(0), query.value(1)])
                    holderR[c, 0, :]=np.array([query.value(2), query.value(3)])
                    holderObj[c, :2]=np.array([query.value(4), query.value(5)])
                    c+=1
                objpoints.append(holderObj.astype('float32'))
                imgpointsL.append(holderL.astype('float32'))
                imgpointsR.append(holderR.astype('float32'))
            ret, mtxL, distL, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpointsL, imsizeL,np.zeros((3,3)),np.zeros((1,5)))
            ret, mtxR, distR, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpointsR, imsizeR,np.zeros((3,3)),np.zeros((1,5)))
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)
            retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F=cv2.stereoCalibrate(objpoints,
                    imgpointsL, imgpointsR,mtxL, distL ,mtxR, distR, imsizeL,np.zeros((3, 1), dtype=np.float64),
                    np.eye(3, dtype=np.float64), criteria=criteria, flags = flags)
            PL = np.hstack((np.eye(3).astype('float32') ,np.zeros((3, 1)).astype('float32')))
            PR = np.hstack((R ,T))

            # check cal
            pscore=[]
            for i in range(len(procInd)):
            #self.loadImagePair(kk+1)

                imgptsL=cv2.undistortPoints(imgpointsL[i], cameraMatrixL, distCoeffsL)
                imgptsR=cv2.undistortPoints(imgpointsR[i], cameraMatrixR, distCoeffsR)
                X=cv2.triangulatePoints(PL, PR, imgptsL,imgptsR)
                X/=X[3]
                ptsL, J=cv2.projectPoints(X[0:3].T,  np.eye(3), np.zeros((3, 1)),cameraMatrixL, distCoeffsL)
                ptsR, J=cv2.projectPoints(X[0:3].T, R, T, cameraMatrixR, distCoeffsR)
                pscore.append(np.vstack((np.array(abs(imgpointsR[i]-ptsR)),np.array(abs(imgpointsL[i]-ptsL)) )).sum())
                if testing:
                    self.loadImagePair(procInd[i])
                    for j in range(ptsL.shape[0]):

                        marker = self.gvLeft.addMark(QPointF(ptsL[j][0][0],  ptsL[j][0][1]), style='+', color=[255,255,0],
                                                      size=1.0, thickness=1.0, alpha=255)
                        #marker.addLabel(str(j), color=[255,255,0], offset=QPointF(-2,-2), name='tnumber',  size=10)
                        self.pointMarks.append(marker)
                        marker = self.gvRight.addMark(QPointF(ptsR[j][0][0],  ptsR[j][0][1]), style='+', color=[255,255,0],
                                                      size=1.0, thickness=1.0, alpha=255)
                        #marker.addLabel(str(j), color=[255,255,0], offset=QPointF(-2,-2), name='tnumber',  size=10)
                        self.pointMarks.append(marker)
                        marker = self.gvLeft.addMark(QPointF(imgpointsL[i][j][0][0],  imgpointsL[i][j][0][1]), style='+', color=[255,0,0],
                                                      size=1.0, thickness=1.0, alpha=255)
                        #marker.addLabel(str(j), color=[225,0,0], offset=QPointF(-2,-2), name='tnumber',  size=10)
                        self.pointMarks.append(marker)
                        marker = self.gvRight.addMark(QPointF(imgpointsR[i][j][0][0],  imgpointsR[i][j][0][1]), style='+', color=[255,0,0],
                                                      size=1.0, thickness=1.0, alpha=255)
                        #marker.addLabel(str(j), color=[225,0,0], offset=QPointF(-2,-2), name='tnumber',  size=10)
                        self.pointMarks.append(marker)
                    return


            badind=np.where(np.array(pscore)>devThresh)
            list=procInd[badind[0].astype('int8')]
            if len(list)>0:
                numstr=''
                for num in list:
                    numstr=numstr+str(num)+' ,'
                reply=QMessageBox.warning(self, "WARNING", "Images "+numstr+" have high inconsitencies.  Rerun the cal without these?",  QMessageBox.Yes, QMessageBox.No)
                if reply==QMessageBox.Yes:
                    imgpointsLsub=[]
                    imgpointsRsub=[]
                    objpointssub=[]
                    for i in range(len(imgpointsL)):
                        if not i in badind[0]:
                            imgpointsLsub.append(imgpointsL[i])
                            imgpointsRsub.append(imgpointsR[i])
                            objpointssub.append(objpoints[i])
                    ret, mtxL, distL, rvecs, tvecs = cv2.calibrateCamera(objpointssub, imgpointsLsub, imsizeL,np.zeros((3,3)),np.zeros((1,5)))
                    ret, mtxR, distR, rvecs, tvecs = cv2.calibrateCamera(objpointssub, imgpointsRsub, imsizeR,np.zeros((3,3)),np.zeros((1,5)))
                    retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F=cv2.stereoCalibrate(objpointssub, imgpointsLsub, imgpointsRsub,mtxL, distL ,mtxR, distR, imsizeL,np.zeros((3, 1), dtype=np.float64), np.eye(3, dtype=np.float64), criteria=criteria, flags = flags)
            # show results
            self.infodlg.show()
            self.infodlg.updateLog('Calibration Results', 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Left Camera Matrix:', 'black')
            self.infodlg.updateLog(cameraMatrixL, 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Left Camera Distortion:', 'black')
            self.infodlg.updateLog(distCoeffsL, 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Right Camera Matrix:', 'black')
            self.infodlg.updateLog(cameraMatrixR, 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Right Camera Distortion:', 'black')
            self.infodlg.updateLog(distCoeffsR, 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Rotation matrix', 'black')
            self.infodlg.updateLog(R, 'black')
            self.infodlg.updateLog('', 'black')
            self.infodlg.updateLog('Translation matrix', 'black')
            self.infodlg.updateLog(T, 'black')
            # grab results
            self.calResults={'cameraMatrixL':cameraMatrixL,'distCoeffsL':distCoeffsL,
           'cameraMatrixR':cameraMatrixR,'distCoeffsR':distCoeffsR,
                'R':R, 'T':T ,  'F':F,  'imageSizeL':imsizeL, 'imageSizeR':imsizeR}


        except Exception as e:
            msg = ''.join(s for s in str(e) if s in string.printable)
            QMessageBox.critical(self, 'Error', msg)


    def saveAndExit(self):
        try:
            dlg=QFileDialog()
            dlg.setAcceptMode(QFileDialog.AcceptSave)
            dlg.setFilter(QDir.Files)
            fileName,  _ = dlg.getSaveFileName(self,'Save Calibration As...', self.__dataDir, "Numpy Data File(*.npz)")
            np.savez(str(fileName), cameraMatrixL=self.calResults['cameraMatrixL'],#,
                distCoeffsL=self.calResults['distCoeffsL'],
                cameraMatrixR=self.calResults['cameraMatrixR'],
                distCoeffsR=self.calResults['distCoeffsR'],
                R=self.calResults['R'],
                T=self.calResults['T'], 
                imageSizeL=self.calResults['imageSizeL'], 
                imageSizeR=self.calResults['imageSizeR']
                )
            self.close()
        except Exception as e:
            msg = ''.join(s for s in str(e) if s in string.printable)
            QMessageBox.warning(self, "ERROR", msg)


    def imageClick(self, imageObj, clickLoc, button, currentKbKey, item):
        try:

            #  check if we're still working on 4 corners for an image
            if (self.cornerCount[imageObj] < 4):

                self.dirty=True

                #  set the view object and image based on which image we're working in
                if (imageObj == self.gvLeft):
                    gvObj=self.gvLeft
                    im=self.grayLeft
                else:
                    gvObj=self.gvRight
                    im=self.grayRight


                if float(self.wtEdit.text())==0:
                    weight=None
                else:
                    weight=1./float(self.wtEdit.text())*100.

                x = int(clickLoc.x())
                y = int(clickLoc.y())
                self.harrisThresh = float(self.harrisEdit.text())
                xBox = int(int(self.xDimEdit.text()) / 2)
                yBox = int(int(self.xDimEdit.text()) / 2)
                if (x < xBox):
                    x = xBox
                if (x > im.shape[1] - xBox):
                    x = im.shape[1]-xBox
                if (y < yBox):
                    y = yBox
                if (y > im.shape[0]-yBox):
                    y = im.shape[0]-yBox

                verts=[]
                verts.append([int(x-xBox), int(y+yBox)])
                verts.append([int(x+xBox), int(y+yBox)])
                verts.append([int(x+xBox), int(y-yBox)])
                verts.append([int(x-xBox), int(y-yBox)])
                point,  score = self.findSingleCorner(im,  clickLoc,  verts, weight)
                if not point:
                    QMessageBox.warning(self, "ERROR", "Unable to detect corner. Are you sure you clicked on a intersection?")
                    return
                marker = gvObj.addPolygon(verts, color=self.marker_color, thickness=3.0, alpha=255, linestyle='=')
                self.pointMarks.append(marker)
                marker = gvObj.addMark(point, style='+', color=self.crosshair_color , size=1.0, thickness=1.0, alpha=255)
                self.cornerPointMarks.append(marker)

                self.cornerPos[imageObj][self.cornerCount[imageObj], ]=np.array([point.x(),point.y()])
                self.cornerCount[imageObj]+=1

                #  check if this is the last corner
                if self.cornerCount[imageObj]==4:
                    # figure out orientation
                    totpoints=[]
                    totObjpoints=[]
                    totfit=[]
                    for rep in range(2):
                        c=0
                        if rep==0:
                            dim1=self.patternDim[1]-1
                            dim2=self.patternDim[0]-1
                        else:
                            dim1=self.patternDim[0]-1
                            dim2=self.patternDim[1]-1


                        temp2 = self.cornerPos[imageObj][:,0].argsort()
                        ranks1 = np.empty(len(temp2), int)
                        ranks1[temp2] = np.arange(len(temp2))
                        ranks1[np.where(ranks1<2)]=0
                        ranks1[np.where(ranks1>1)]=1
                        temp2 = self.cornerPos[imageObj][:,1].argsort()
                        ranks2 = np.empty(len(temp2), int)
                        ranks2[temp2] = np.arange(len(temp2))
                        ranks2[np.where(ranks2<2)]=0
                        ranks2[np.where(ranks2>1)]=1
                        ul=np.where(np.logical_and(ranks1==0,ranks2==0))
                        ur=np.where(np.logical_and(ranks1==1,ranks2==0))
                        lr=np.where(np.logical_and(ranks1==1,ranks2==1))
                        ll=np.where(np.logical_and(ranks1==0,ranks2==1))
                        corners=np.zeros((4, 2))
                        corners[0, :]=self.cornerPos[imageObj][ul, ]
                        corners[1, :]=self.cornerPos[imageObj][ur, ]
                        corners[2, :]=self.cornerPos[imageObj][lr, ]
                        corners[3, :]=self.cornerPos[imageObj][ll, ]

                        pts=np.array([[corners[0,0],    corners[0,1]],
                        [corners[1,0],  corners[1,1]],
                        [corners[2,0],  corners[2,1]],
                        [corners[3,0],  corners[3,1]]])
                        mod=np.array([[0.,  0.],
                        [1.,    0.],
                        [1.,    1.],
                        [0.,    1.]])
                        mat=cv2.findHomography(mod,  pts)
                        self.dest=np.zeros((3, (dim1+1)*(dim2+1)))
                        c1=0
                        c2=0
                        for i in range(int(self.dest.shape[1])):
                            self.dest[:, i]=np.array([c2,c1,  0])
                            if c2==dim1:
                                c2=0
                                c1+=1
                            else:
                                c2+=1

                        modPts=copy.deepcopy(self.dest)
                        modPts[0, :]=modPts[0, :]/float(dim1)
                        modPts[1, :]=modPts[1, :]/float(dim2)
                        modPts[2, :]=modPts[2, :]+1

                        XX=np.dot(mat[0], modPts)
                        p=np.array([XX[2,:]])
                        res = XX[0:2,:] / np.dot(np.ones((2,1)), p)
                        deltax=abs(corners[0,0]-corners[1,0])
                        xSubBox=int(max(round(np.around((deltax/float(dim1))*.05)), int(self.xDimEdit.text())))
                        ySubBox=xSubBox
                        holder=np.zeros(((dim2+1)*(dim1+1), 2))
                        fit=np.zeros(((dim2+1)*(dim1+1), 1))
                        for pt in res.T:

                            xt=int(round(pt[0]))
                            yt=int(round(pt[1]))
                            verts=[]
                            verts.append([xt-xSubBox, yt+ySubBox])
                            verts.append([xt+xSubBox, yt+ySubBox])
                            verts.append([xt+xSubBox, yt-ySubBox])
                            verts.append([xt-xSubBox, yt-ySubBox])
                            point, score =self.findSingleCorner(im,  QPointF(yt, xt),  verts,weight )
                            if not point:
                                QMessageBox.warning(self, "ERROR", "having trouble extracting sub corners.  You might try changing the settings some?")
                                return

                            holder[c]=np.array([point.x(),  point.y()])
                            fit[c]=score
                            c+=1
                        totfit.append(fit.sum())
                        totpoints.append(holder)
                        totObjpoints.append(self.dest.T*self.scale)

                    if totfit[0]>totfit[1]:
                        choicePoints=totpoints[0]
                        objPoints=totObjpoints[0]
                    else:
                        choicePoints=totpoints[1]
                        objPoints=totObjpoints[1]

                    for mark in self.cornerPointMarks:
                        imageObj.removeItem(mark)
                    self.cornerPointMarks=[]
                    c=1
                    for point in choicePoints:
                        marker = gvObj.addMark(QPointF(point[0],  point[1]), style='+', color=self.crosshair_color,
                                                      size=1.0, thickness=1.0, alpha=255)
                        marker.addLabel(str(c), color=self.text_color, offset=QPointF(-2,-2),  size=self.text_size)
                        self.pointMarks.append(marker)
                        c+=1
                    self.chessPoints[imageObj]=choicePoints
                    self.objectPoints=objPoints

        except Exception as e:
            msg = ''.join(s for s in str(e) if s in string.printable)
            QMessageBox.critical(self, 'Error', msg)


    def findSingleCorner(self,  im,  clickLoc,  verts,  sigma=None):

        try:
            rowArray=np.linspace(verts[3][1], verts[0][1]-1, verts[0][1]-verts[3][1])
            colArray=np.linspace(verts[0][0], verts[1][0]-1, verts[1][0]-verts[0][0])
            chip=im[verts[3][1]:verts[0][1]][:, verts[0][0]:verts[1][0]]
            im2=cv2.cornerHarris(chip,3,5,0.04)
            # make gaussian kernel
            if sigma:
                kernel1=cv2.getGaussianKernel(len(rowArray), sigma)
                kernel2=cv2.getGaussianKernel(len(colArray), sigma)
                kernel=np.dot(kernel1, kernel2.T)
                kernel=kernel/kernel.max()
                im2=im2*kernel
            l=np.where(im2>self.harrisThresh*im2.max())
#            chip[l]=255
#            cv2.imshow('img',chip)
#            cv2.waitKey()
            if l[0].shape[0]>0:
                ind=[int(round(np.mean(l[0]))), int(round(np.mean(l[1])))]
                pt=QPointF(colArray[ind[1]], rowArray[ind[0]])
                score=im2[l].mean()
            else:
                pt=clickLoc
                score=0
            return pt, score
        except Exception as e:
            msg = ''.join(s for s in str(e) if s in string.printable)
            QMessageBox.critical(self, 'Error', msg)
            return False


    def autoFindCorners(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # Find the chess board corners
        if not self.autoMarks[0]:
            leftOK, leftCorners = cv2.findChessboardCorners(self.grayLeft, (self.patternDim[0]+1,
                    self.patternDim[1]+1), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if leftOK == True:
                self.drawCorners(self.gvLeft, leftCorners)
                self.autoMarks[0]=True
            else:
                QMessageBox.warning(self, "ERROR", "couldn't find left corners")
        if not self.autoMarks[1]:
            rightOK, rightCorners = cv2.findChessboardCorners(self.grayRight, (self.patternDim[0]+1,
                    self.patternDim[1]+1), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if rightOK == True:
                self.drawCorners(self.gvRight, rightCorners)
            else:
                QMessageBox.warning(self, "ERROR", "couldn't find right corners")
        QApplication.restoreOverrideCursor()


    def drawCorners(self,  gvObj,  corners):
        self.clearMarks()
        style='+'
        for i, corner in enumerate(corners):
            marker = gvObj.addMark(QPointF(corner[0][0], corner[0][1]), style=style, color=self.crosshair_color,
                                                  size=1.0, thickness=1.0, alpha=255)
            marker.addLabel(str(i+1), color=self.text_color, offset=QPointF(-2,-2),  size=self.text_size)


    def startNewCal(self):
        '''
        startNewCal is called when the "Start Cal" button is pressed. It creates the database file,
        sorts out the calibration images and loads the first pair.
        '''
        dlg=imgfiledlg.ImageFileDlg(self)
        ok = dlg.exec_()
        
        if not ok:
            return
            
        lBase=str(dlg.lBaseEdit.text())
        rBase=str(dlg.rBaseEdit.text())
        type=str(dlg.typeBox.currentText())
        self.patternDim=[int(dlg.rowNumEdit.text()),int(dlg.colNumEdit.text())]
        self.scale=float(dlg.scaleEdit.text())

        name=dlg.nameEdit.text()
        if name=='':
            name='StereoCalibration'
        self.workingPath = str(dlg.pathEdit.text())

        #  update the internal cal image filename data structs
        ok = self.updateImageFiles(lBase, rBase, type)
        if (not ok):
            return

        # check if a cal database file exists

        if QFile(self.workingPath + os.sep + 'StereoCalPoints.sql3').exists():
            reply = QMessageBox.warning(self, "WARNING", "You've been working on this cal. Do you want to " +\
                    "start all over from scratch? If not, click 'No' and use the reload button.",
                    QMessageBox.Yes|QMessageBox.No)
            if (reply == QMessageBox.No):
                #  user doens't want to replace it
                return
            else:
                #  user wants to replace the existing cal so delete it
                os.remove(self.workingPath + os.sep + 'StereoCalPoints.sql3')

        #  create the calibration database
        self.db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(self.workingPath + os.sep + 'StereoCalPoints.sql3')

        #  and try to open
        if not self.db.open():
            QMessageBox.critical(self, "ERROR", "Unable to create the calibration database. Do you " +
                    "have write permissions in the calibration folder?")
            return

        #  create the "points" table
        query=QtSql.QSqlQuery("create table points(frame_number INTEGER, " +
                " left_image_filename TEXT, right_image_filename TEXT,point_number NUMERIC," +
                "chess_pt_LX NUMERIC, chess_pt_LY  NUMERIC, chess_pt_RX  NUMERIC, " +
                "chess_pt_RY  NUMERIC, object_pt_X NUMERIC, object_pt_Y NUMERIC,object_pt_Z NUMERIC)")
        query.exec_()

        #  create the "cal_params" table
        query=QtSql.QSqlQuery("create table cal_params(parameter TEXT, parameter_value TEXT)")
        query.exec_()

        #  insert parameters into table
        params=['lBase', 'rBase', 'type', 'rowNum', 'colNum', 'square_scale', 'image_directory']
        vals=[lBase, rBase, type, dlg.rowNumEdit.text(), dlg.colNumEdit.text(), dlg.scaleEdit.text(), dlg.pathEdit.text()]
        for i, param in enumerate(params):
            query = QtSql.QSqlQuery("INSERT INTO cal_params (parameter,  parameter_value) VALUES('" +
                    param+"', '"+vals[i]+"')")

        #  load up the first image pair
        self.num=0
        self.loadImagePair(self.shotList[self.num])
        self.totLabel.setText(str(len(self.shotList)))


    def reloadCal(self):
        '''
        reloadCal reloads an existing cal.
        '''

        #  get the path to the existing calibration directory
        dirDlg = QFileDialog(self)
        workingPath = str(dirDlg.getExistingDirectory(self, 'Select Calibration Image Directory',
                self.__dataDir,QFileDialog.ShowDirsOnly))
        if (workingPath == ''):
            return
        self.workingPath = workingPath

        #  open the calibration database
        self.db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(self.workingPath + os.sep + 'StereoCalPoints.sql3')
        if not self.db.open():
            QMessageBox.critical(self, "Fatal", "Unable to open the calibration database.")
            return

        # get params
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='lBase'")
        query.first()
        lBase=str(query.value(0))
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='rBase'")
        query.first()
        rBase=str(query.value(0))
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='type'")
        query.first()
        type=str(query.value(0))

        self.patternDim=[]
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='rowNum'")
        query.first()
        self.patternDim.append(int(query.value(0)))
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='colNum'")
        query.first()
        self.patternDim.append(int(query.value(0)))
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='square_scale'")
        query.first()
        self.scale=float(query.value(0))
        query=QtSql.QSqlQuery("SELECT parameter_value FROM cal_params WHERE parameter='image_directory'")
        query.first()
        #self.workingPath=str(query.value(0))

        #  update the internal cal image filename data structs
        ok = self.updateImageFiles(lBase, rBase, type)
        if (not ok):
            return

        query=QtSql.QSqlQuery("SELECT max(frame_number) FROM points")
        self.num=1
        if query.first():
            if query.value(0) in self.shotList:
                self.num=self.shotList.index(query.value(0))+1
        if not (self.num)>=len(self.leftImages):
            self.loadImagePair(self.shotList[self.num])
        else:
            QMessageBox.warning(self, "INFO", "Finished with image set.")
            self.loadImagePair(self.shotList[self.num-1])
        self.totLabel.setText(str(len(self.shotList)))


    def scrubImage(self):
        self.loadImagePair(self.shotList[self.num])


    def loadImagePair(self, num):
        self.clearMarks()

        for obj in [self.gvLeft, self.gvRight]:
            #   this is brute force, but works around an c++/Python object deleting issue
            obj.removeScene()
            obj.createScene()

        self.setImages(num)
        self.frameLabel.setText('Image - '+str(num))
        self.reloadFrameData(num)


    def updateImageFiles(self, lBase, rBase, type):
        '''
        updateImageFiles updates the internal data structs that hole the calibration image info
        '''
        #  build a dict of image filenames
        leftImages = glob.glob(self.workingPath + os.sep + lBase + '*' + type)
        rightImages = glob.glob(self.workingPath + os.sep + rBase + '*' + type)
        self.leftImages={}
        self.rightImages={}
        self.shotList=[]

        #  work thru the left images
        regEx = '(?!' + lBase + ')\d+'
        for file in leftImages:
            path, filename = os.path.split(file)
            filename , extension = os.path.splitext(filename)
            try:
                imageNum = re.findall(regEx,filename)[0]
                imageNum = int(imageNum)
                self.leftImages.update({imageNum:file})
                self.shotList.append(imageNum)
            except:
                QMessageBox.critical(self, "Error!", "Unable to parse image name. Images must be named " +
                        "<base name><integer image number>.<extension>")
                return False
        #  and now the right images
        regEx = '(?!' + rBase + ')\d+'
        for file in rightImages:
            path, filename = os.path.split(file)
            filename , extension = os.path.splitext(filename)
            try:
                imageNum = re.findall(regEx,filename)[0]
                imageNum = int(imageNum)
                self.rightImages.update({imageNum:file})
            except:
                QMessageBox.critical(self, "Error!", "Unable to parse image name. Images must be named " +
                        "<base name><integer image number>.<extension>")
                return False

        #  sort the shot list so images are in numerical order, not as the OS sorts the files
        self.shotList.sort()

        return True


    def reloadFrameData(self, num):
        self.dirty=False
        self.cornerPos[self.gvLeft]=np.zeros([4, 2])
        self.cornerPos[self.gvRight]=np.zeros([4, 2])
        self.cornerCount[self.gvLeft]=0
        self.cornerCount[self.gvRight]=0
        # how many points?
        query=QtSql.QSqlQuery("SELECT count(*) FROM points WHERE frame_number="+str(num))
        query.first()
        if query.value(0)>0:
            self.chessPoints[self.gvLeft]=np.zeros((query.value(0), 2))
            self.chessPoints[self.gvRight]=np.zeros((query.value(0), 2))
            self.objectPoints=np.zeros((query.value(0), 3))
            query=QtSql.QSqlQuery("SELECT chess_pt_LX, chess_pt_LY , chess_pt_RX, "+
                        "chess_pt_RY, object_pt_X, object_pt_Y,object_pt_Z FROM points WHERE frame_number="+str(num)+" ORDER BY point_number")
            self.cornerCount[self.gvLeft]=4
            self.cornerCount[self.gvRight]=4
            c=0
            while query.next():
                self.chessPoints[self.gvLeft][c, 0]=query.value(0)
                self.chessPoints[self.gvLeft][c, 1]=query.value(1)
                marker = self.gvLeft.addMark(QPointF(query.value(0), query.value(1)), style='+',
                        color=self.crosshair_color, size=1.0, thickness=1.0, alpha=255)
                marker.addLabel(str(c+1), color=self.text_color, offset=QPointF(-2,-2),  size=self.text_size)
                self.cornerPointMarks.append(marker)
                self.chessPoints[self.gvRight][c, 0]=query.value(2)
                self.chessPoints[self.gvRight][c, 1]=query.value(3)
                marker = self.gvRight.addMark(QPointF(query.value(2), query.value(3)), style='+',
                        color=self.crosshair_color, size=1.0, thickness=1.0, alpha=255)
                marker.addLabel(str(c+1), color=self.text_color, offset=QPointF(-2,-2),  size=self.text_size)
                self.cornerPointMarks.append(marker)
                self.objectPoints[c, 0]=query.value(4)
                self.objectPoints[c, 1]=query.value(5)
                self.objectPoints[c, 2]=query.value(6)
                c+=1




    def setImages(self,  num):
        '''
        setImages loads a pair of images and optionally applies some simple operations
        to "clean them up".

        This should possibly be re-worked since at least the contrast/brightness business
        is already handled by the QImageviewer class (QEnhancedImage).
        '''
        try:
            self.gvLeft.setImageFromFile(self.leftImages[num])
            self.gvLeft.resetView()
            self.gvLeft.fillExtent()
            self.gvRight.setImageFromFile(self.rightImages[num])
            self.gvRight.resetView()
            self.gvRight.fillExtent()
            self.left = cv2.imread(self.leftImages[num])
            grayLeft = cv2.cvtColor(self.left, cv2.COLOR_BGR2GRAY)
            self.grayLeft = self.clahe.apply(grayLeft)
            if self.scrubBox.isChecked():
                self.grayLeft = cv2.morphologyEx(self.grayLeft, cv2.MORPH_OPEN, self.kernel)
                self.grayLeft = cv2.morphologyEx(self.grayLeft, cv2.MORPH_CLOSE, self.kernel)
            self.right = cv2.imread(self.rightImages[num])
            grayRight = cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY)
            self.grayRight = self.clahe.apply(grayRight)
            if self.scrubBox.isChecked():
                self.grayRight = cv2.morphologyEx(self.grayRight, cv2.MORPH_OPEN, self.kernel)
                self.grayRight = cv2.morphologyEx(self.grayRight, cv2.MORPH_CLOSE, self.kernel)

        except Exception as e:
            msg = ''.join(s for s in str(e) if s in string.printable)
            QMessageBox.warning(self, "ERROR", msg)


    def clearPoints(self):
        '''
        clearPoints clears all marks and points from the specified image.
        '''
        #  check which button (left/right) was clicked and clear that image
        if self.sender()==self.clearLeftBtn:
            self.clearMarks('Left')
            self.cornerPos[self.gvLeft] = np.zeros([4, 2])
            self.cornerCount[self.gvLeft] = 0
            self.chessPoints[self.gvLeft] = np.array([])
        elif self.sender()==self.clearRightBtn:
            self.clearMarks('Right')
            self.cornerPos[self.gvRight] = np.zeros([4, 2])
            self.cornerCount[self.gvRight] = 0
            self.chessPoints[self.gvRight] = np.array([])


    def clearMarks(self,  img=None):
        '''
        clearMarks clears the marks the user placed on an image from the specified image. If no
        image is specified the marks are cleared from both.
        '''
        if (img):
            if (img.lower() == 'left'):
                self.gvLeft.removeAllItems()
            else:
                self.gvRight.removeAllItems()
        else:
            self.gvLeft.removeAllItems()
            self.gvRight.removeAllItems()

        self.pointMarks = []
        self.cornerPointMarks = []


    def testCal(self):
        '''
        testCal seems to read in an existing cal then cxompute the size of the calibration
        target chess board square based on the cal and then writes it to an xls spreadsheet.
        This is a bit of a mess, and should be cleaned up. For now it is here as an example of
        how one could test the "quality" of their calibration.
        '''
        import xlrd,  xlwt,  xlutils.copy
        workbook = xlwt.Workbook(encoding = 'ascii')
        worksheet = workbook.add_sheet('CalTest')
        import pyStereoComp

        compset=pyStereoComp.pyStereoComp()
        query=QtSql.QSqlQuery("SELECT frame_number, count(*) FROM points GROUP BY frame_number ORDER BY frame_number")
        first=True
        unit=10
        row=0
        while query.next():
            col=0
            query1=QtSql.QSqlQuery("SELECT chess_pt_LX, chess_pt_LY , chess_pt_RX, "+
                    "chess_pt_RY FROM points WHERE frame_number="+query.value(0))

            worksheet.write(row, col, str(query.value(0)))
            while query1.next():

                xLc=np.array([[query1.value(0)], [ query1.value(1)]])
                xRc=np.array([[query1.value(2)], [  query1.value(3)]])
                if not first:
                    XLp, XRp=compset.triangulatePoint(xLp, xRp)
                    XLc, XRc=compset.triangulatePoint(xLc, xRc)
                    lhx=XLc[0, 0]/unit
                    lhy=XLc[2, 0]/unit
                    lhz=-XLc[1, 0]/unit
                    ltx=XLp[0, 0]/unit
                    lty=XLp[2, 0]/unit
                    ltz=-XLp[1, 0]/unit
                    col+=1
                    #length.append(round(sqrt((lhx-ltx)**2+(lhy-lty)**2+(lhz-ltz)**2), len(str(unit))))
                    worksheet.write(row, col, str(round(sqrt((lhx-ltx)**2+(lhy-lty)**2+(lhz-ltz)**2), len(str(unit)))))

                xLp=xLc
                xRp=xRc
                first=False
            #print(query.value(0))
            row+=1
        workbook.save(str(self.workingPath + os.sep + 'Caltest.xls'))


    def closeEvent(self,  event):

        #  save the application state
        self.__appSettings.setValue('winposition', self.pos())
        self.__appSettings.setValue('winsize', self.size())
        self.__appSettings.setValue('corner_finder_dim',self.xDimEdit.text())
        self.__appSettings.setValue('click_weight',self.wtEdit.text())
        self.__appSettings.setValue('harris_thresh',self.harrisEdit.text())
        self.__appSettings.setValue('dev_thresh',self.devThreshEdit.text())
        self.__appSettings.setValue('datadir', self.workingPath)

        #  close the database
        if self.db != None:
            self.db.close()

        self.infodlg.hide()
        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    form = StereoCalibrate()
    form.show()
    app.exec_()

