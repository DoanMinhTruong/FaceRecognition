from PyQt5 import QtGui
from PyQt5.QtWidgets import QStatusBar,QFileDialog, QWidget, QApplication, QLabel, QGridLayout
from PyQt5 import QtWidgets
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDir, Qt, QUrl, QSize, QTimer
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from face import DracoFaceRecognition
import json
import pickle
from datetime import datetime
import requests
# dfr = DracoFaceRecognition()
dfr = pickle.load( open("model.pickle" , 'rb'))
faceDetect = cv2.CascadeClassifier('haar.xml')


data = json.load(open('data.json'))['data']


DATETIME_FORMAT = "%H:%M:%S %d/%m/%y"
WORKING_TIME = 8


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self._run_flag = True

    def start(self):
        super().start()
        print('start')
        self._run_flag = True
    
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0 + cv2.CAP_MSMF)
        while self._run_flag:
            try:
                ret, cv_img = cap.read()
                if ret:
                    cv_img = cv2.flip(cv_img , 1)
                    # centerH = cv_img.shape[0] // 2;
                    # centerW = cv_img.shape[1] // 2;
                    # sizeboxW = 300;
                    # sizeboxH = 400;
                    # cv2.rectangle(cv_img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                    #             (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
                    
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    faces=faceDetect.detectMultiScale(gray,1.3,5)
                    for(x,y,w,h) in faces:
                        cv2.rectangle(cv_img,(x,y),(x+w,y+h),(255,0,0),2)

                    self.change_pixmap_signal.emit(cv_img)
            except:
                print("Error")
        cap.release()
        cv2.destroyAllWindows()
    # def run_reg(self):
    #     cap = cv2.VideoCapture(0)
    #     while self._run_flag:
    #         ret, cv_img = cap.read()
    #         if ret:
    #             cv_img = cv2.flip(cv_img , 1)
    #             centerH = cv_img.shape[0] // 2;
    #             centerW = cv_img.shape[1] // 2;
    #             sizeboxW = 300;
    #             sizeboxH = 400;
    #             cv2.rectangle(cv_img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
    #                         (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
                
    #             gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    #             faceLoc, faceName = dfr.detect_known_faces(gray)
    #             loc = (np.unique(faceLoc)).tolist()
    #             name = (np.unique(faceName)).tolist()
    #             # # print(loc, name)
    #             if(len(loc)):
    #                 # y1,x1,y2,x2= loc[0] , loc[1] , loc[2] , loc[3]
    #                 r = name[0].split('.')[0]
    #                 info = None
    #                 for i in data:
    #                     if(i['id'] == r):
    #                         info = i
    #                         break
    #                 global res
    #                 res = info
    #             if(res):
    #                 key = cv2.waitKey(-1)
    #             # faces=faceDetect.detectMultiScale(gray,1.3,5)
    #             # for(x,y,w,h) in faces:
    #                 # cv2.rectangle(cv_img,(x,y),(x+w,y+h),(255,0,0),2)
    #                 self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        
    def stop(self):
        self._run_flag = False
        self.wait()

def video_to_image(video = None , name = None):
        enough = False
        if(video):
            name = (video.split("/")[-1]).split(".")[0]
            video = cv2.VideoCapture(video)
        else:
            name = name
            video = cv2.VideoCapture(0 + cv2.CAP_MSMF)
        count = 0 
        success, image = video.read()
        
        images = []
        frame_step = 0
        while success:
            try:
                success,image = video.read()
                if(frame_step <=15):
                    frame_step += 1
                    continue
                if(count == 14 or count == 15):
                    break
                frame_step = 0
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                faces=faceDetect.detectMultiScale(gray,1.3,5)
                for(x,y,w,h) in faces:
                    fn = name + "." + str(count) 
                    count +=1
                    fn_flip = name + "." + str(count)
                    crop = gray[y:y+h,x:x+w]
                    crop_flip = cv2.flip(crop , 1)

                    images.append({'image' : crop , 'filename' : fn})
                    print("pre " + fn)

                    images.append({'image' : crop_flip , 'filename' : fn_flip})
                    print("pre " + fn_flip)

                    count += 1
                # if(count == 201 or count == 202):
                #     enough = True
                #     break
                # print("pre " + name)
                
            except:
                print("Error at " , video , count)
        video.release()
        cv2.destroyAllWindows()
        dfr.load_encoding_images(images)


class LiveStream(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveStream")
        self.setFixedWidth(640)
        self.setFixedHeight(700)
        self.start = QtWidgets.QPushButton(self)
        self.start.setText('Start')
        self.start.clicked.connect(self.call_video_to_image)

        self.display_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        
        self.vbox = QGridLayout(self)
        self.vbox.addWidget(self.image_label)
        self.setLayout(self.vbox)
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
    def call_video_to_image(self):
        video_to_image(None , 'truong')
    def show(self):
        # self.hide()
        self.thread.start()
        super().show()
        print("show")
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
class Input(QWidget):
    def __init__(self):
        super().__init__()
        

        self.setWindowTitle("Input Data")
        self.setFixedWidth(300)
        self.setFixedHeight(400)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)
        self.livestream = QtWidgets.QPushButton(self)
        self.livestream.setText('Live Import')
        self.livestream.clicked.connect(self.openLive)
        
        

        
        self.add_import_video()

        vbox = QGridLayout()
        vbox.addWidget(self.button_add_import_video)
        vbox.addWidget(self.livestream)

        self.setLayout(vbox)
        # self.thread = VideoThread()
        # create the video capture thread
        # self.thread.change_pixmap_signal.connect(self.update_image)
    def openLive(self):
        self.live = LiveStream()
        self.live.show() 
    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())
   
  
    def import_video(self):
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Video Choosing",
            ".", "Video Files (*.mp4 *mov *wmv *avchd *webm *mkv *.flv *.ts *.mts *.avi)")
        for fileName in fileNames:
            if fileName == '':
                return
        
        for fileName in fileNames:
            video_to_image(fileName)

    def add_import_video(self):
        self.button_add_import_video = QtWidgets.QPushButton(self)
        self.button_add_import_video.setText("Import Video")
        self.button_add_import_video.clicked.connect(self.import_video)
    def closeEvent(self, event):
        main.show()
        pickle.dump(dfr , open('model.pickle' , 'wb'))
        event.accept()
    
    def show(self):
        self.hide()
        super().show()

    
    
class Checkin(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Checkin")
        self.setFixedWidth(900)
        self.setFixedHeight(500)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.display_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        
        self.result = QLabel("")
        font = QFont('Consolas' , 13)
        self.result.setFont(font)
        
        self.submit = QtWidgets.QPushButton(self)
        self.submit.setText("Submit")
        self.submit.clicked.connect(self.saveResult)
        self.submit.setEnabled(False)
        self.submit.setFixedSize(QSize(80, 19))

        self.reset  = QtWidgets.QPushButton(self)
        self.reset.setText("Reset")
        self.reset.clicked.connect(self.handle_reset)
        self.reset.setFixedSize(QSize(80, 19))


        vbox = QGridLayout()
        vbox.addWidget(self.image_label, 0 , 0, 4 ,3)
        vbox.addWidget(self.statusBar , 4 , 0 , 1, 1)
        vbox.addWidget(self.result, 0 , 4)

        vbox.addWidget(self.submit, 3 , 3 )
        vbox.addWidget(self.reset , 3 , 4)

        self.hide_result_json = QLabel(self)
        self.hide_result_json.hide()
        

        self.setLayout(vbox)
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        
    def handle_reset(self):
        self.thread.start()
        self.result.setText('')
        self.submit.setEnabled(False)
    def saveResult(self):
        path = 'checkin.json'
        file = open(path , 'r')
        data = json.load(file)

        data['checkin'].append(json.loads(self.hide_result_json.text()))
        print(data)
        with open(path , 'w') as f:
            json.dump(data, f)
        
        file.close()
        self.handle_reset()
        
    
    def closeEvent(self, event):
        main.show()
        self.thread.stop()
        event.accept()
    
    def show(self):
        super().show()
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        # global res
        # if(res != None):
        #     info = data[res]
        #     self.result = QLabel(f'Id : {info["id"]} <br>Full Name: {info["full_name"]}<br>Birth: {info["birth"]}<br>Gender: {info["gender"]}<br>')
            # res = None        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_SPACE:
            self.close()


    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        faceLoc, faceName = dfr.detect_known_faces(gray)
        loc = (np.unique(faceLoc)).tolist()
        name = (np.unique(faceName)).tolist()
        if(len(loc)):
            # self.thread.stop()
            # self.submit.setEnabled(True)
            # y1,x1,y2,x2= loc[0] , loc[1] , loc[2] , loc[3]
            r = name[0].split('.')[0]
            info = None
            for i in data:
                if(i['id'] == r):
                    info = i
                    break
            if not info:
                info = data[0]
            else:
                self.thread.stop()
                self.submit.setEnabled(True)

            now = datetime.now()

            current_time = now.strftime(DATETIME_FORMAT)
            info['time'] = current_time

            self.hide_result_json.setText(json.dumps(info))
            self.result.setText(f'Id : {info["id"]} \nFull Name: {info["full_name"]}\nBirth: {info["birth"]}\nGender: {info["gender"]}\nTime:{info["time"]}\n')
       
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Checkout(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Checkout")
        self.setFixedWidth(900)
        self.setFixedHeight(500)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.display_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        
        self.result = QLabel("")
        font = QFont('Consolas' , 13)
        self.result.setFont(font)
        
        self.submit = QtWidgets.QPushButton(self)
        self.submit.setText("Submit")
        self.submit.clicked.connect(self.saveResult)
        self.submit.setEnabled(False)
        self.submit.setFixedSize(QSize(80, 19))

        self.reset  = QtWidgets.QPushButton(self)
        self.reset.setText("Reset")
        self.reset.clicked.connect(self.handle_reset)
        self.reset.setFixedSize(QSize(80, 19))

        self.hide_result_json = QLabel(self)
        self.hide_result_json.hide()

        vbox = QGridLayout()
        vbox.addWidget(self.image_label, 0 , 0, 4 ,3)
        vbox.addWidget(self.statusBar , 4 , 0 , 1, 1)
        vbox.addWidget(self.result, 0 , 4)

        vbox.addWidget(self.submit, 3 , 3 )
        vbox.addWidget(self.reset , 3 , 4)

        
        self.setLayout(vbox)
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        
    def handle_reset(self):
        self.thread.start()
        self.result.setText('')
        self.hide_result_json.setText('')
        self.submit.setEnabled(False)
    def saveResult(self):
        path = 'checkout.json'
        file = open(path , 'r')
        data = json.load(file)

        data['checkout'].append(json.loads(self.hide_result_json.text()))
        print(data)
        with open('checkout.json' , 'w') as f:
            json.dump(data, f)
        
        file.close()
        self.handle_reset()
    
    def closeEvent(self, event):
        main.show()
        self.thread.stop()
        event.accept()
    
    def show(self):
        super().show()
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        # global res
        # if(res != None):
        #     info = data[res]
        #     self.result = QLabel(f'Id : {info["id"]} <br>Full Name: {info["full_name"]}<br>Birth: {info["birth"]}<br>Gender: {info["gender"]}<br>')
            # res = None        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_SPACE:
            self.close()


    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        faceLoc, faceName = dfr.detect_known_faces(gray)
        loc = (np.unique(faceLoc)).tolist()
        name = (np.unique(faceName)).tolist()
        # print(name)
        if(len(loc)):
            # self.thread.stop()
            # self.submit.setEnabled(True)
            # y1,x1,y2,x2= loc[0] , loc[1] , loc[2] , loc[3]
            r = name[0].split('.')[0]
            info = None
            for i in data:
                if(i['id'] == r):
                    info = i
                    break
            if not info:
                info = data[0]
            else:
                self.thread.stop()
                self.submit.setEnabled(True)

            now = datetime.now()

            current_time = now.strftime(DATETIME_FORMAT)
            info['time'] = current_time

            self.hide_result_json.setText(json.dumps(info))
            self.result.setText(f'Id : {info["id"]} \nFull Name: {info["full_name"]}\nBirth: {info["birth"]}\nGender: {info["gender"]}\nTime:{info["time"]}\n')

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo Draco")
        self.setFixedWidth(250)
        self.setFixedHeight(250)
        self.input_window = Input()
        self.checkin = Checkin()
        self.checkout = Checkout()
        self.add_button()
        vbox = QGridLayout()
        self.widgets = [self.button_input , self.button_checkin , self.button_checkout ]
        for i in self.widgets:
            vbox.addWidget(i)
        self.setLayout(vbox)
    def open_input(self):
        try:
            self.hide()
            self.input_window.show()
        except:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Oh no!')
    def open_checkin(self):
        try:
            self.hide()
            self.checkin.show()
        except:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Oh no!')
    def open_checkout(self):
        try:
            self.hide()
            self.checkout.show()
        except:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Oh no!')
    def add_button(self):
        self.button_input = QtWidgets.QPushButton(self)
        self.button_input.setText("Input Data")
        self.button_input.clicked.connect(self.open_input)

        self.button_checkin = QtWidgets.QPushButton(self)
        self.button_checkin.setText("Checkin")
        self.button_checkin.clicked.connect(self.open_checkin)

        self.button_checkout = QtWidgets.QPushButton(self)
        self.button_checkout.setText("Checkout")
        self.button_checkout.clicked.connect(self.open_checkout)
    def closeEvent(self, event):
        main.show()
        # self.thread.stop()
        event.accept()
        calculate_end_day()
class Verify(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Verify Check-in")
        self.setFixedWidth(750)
        self.setFixedHeight(500)
        vbox = QGridLayout()
        self.display_width = 640
        self.display_height = 480

        self.info = QLabel('')
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        # self.image_label.setPixmap(main.face_recognition_window.image_label)

        vbox.addWidget(self.image_label , 0 , 0)
        self.setLayout(vbox)
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        # global res
        # if(res != None):
        #     info = data[res]
        #     self.result = QLabel(f'Id : {info["id"]} <br>Full Name: {info["full_name"]}<br>Birth: {info["birth"]}<br>Gender: {info["gender"]}<br>')
            # res = None        
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
app = QApplication(sys.argv)
main = Main()

def result_to_json(text):
    text = text.replace('Id' , '"id"').replace("Name" , '"name"').replace("Birth", '"birth"').replace("Gender" , '"gender"' ).replace("Time" , '"time"')
    return ("{" + text + "},")

def calculate_end_day():
    checkin = open('checkin.json' , 'r')
    checkout = open('checkout.json' , 'r')
    # try:
    te_checkin = json.load(checkin)
    te_checkout = json.load(checkout)

    result = []
    print(te_checkin , te_checkout)
    for i in te_checkin['checkin']:
        for j in te_checkout['checkout']:
            if(i['id'] == j['id']):
                # status = ""
                time_in = datetime.strptime(i['time'] , DATETIME_FORMAT)
                time_out = datetime.strptime(j['time'], DATETIME_FORMAT)
                result.append({
                    'id' : i['id'],
                    'status' : 'present',
                    'check_in' : time_in.strftime(DATETIME_FORMAT),
                    'check_out' : time_out.strftime(DATETIME_FORMAT)  , 
                    'total_hour' : abs(time_out.hour - time_in.hour),
                    'total_minutes' : abs(time_out.minute - time_in.minute),
                    'date' :  (str(time_in.year)) + "-" + (str(time_in.month)) + "-" + (str(time_in.day)),
                })
    print('ok')
    with (open('result.json' , 'w')) as f:
        json.dump({'result' : result}, f)
    # except:
    #     print("Something went wrong! Check your Checkin - Checkout")


    checkin.close()
    checkout.close()

def send_unmarked_day_to_save():
    url = "http://dev3.draco.net:8000/api/method/erpnext.hr.doctype.attendance.attendance.mark_bulk_attendance"
    result = json.load(open('result.json'))['result']
    request = []
    for r in result:
        unmarked = {
            'doctype' : 'Attendance' , 
            'attendance_date' : r['date'],
            'status' : 'Present',
            'company' : 'Fintech DRACO' , 
            'employee' : r['id']
        }
        request.append(unmarked)
    data = {'unmarked_days' : request}
    print(data)
    r = requests.get(url = url , params = data)
    print(r.json())
    # 'doctype': 'Attendance',
	# 		'employee': data.employee,
	# 		'attendance_date': get_datetime(date),
	# 		'status': data.status,
	# 		'company': company,


if __name__=="__main__":
    # main.show()
    # calculate_end_day()
    # send_unmarked_day_to_save()
    sys.exit(app.exec_())
    