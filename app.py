from PIL import Image, ImageTk
import tkinter as tk
import cv2
from keras.models import model_from_json
import operator
from string import ascii_uppercase



class Application:
    def __init__(self):
        self.directory = 'model/'
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.json_file = open(self.directory + "model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory + "model-bw.h5")

        self.json_file_dru = open(self.directory + "model-bw_dru.json", "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(self.directory + "model-bw_dru.h5")

        self.json_file_tkdi = open(self.directory + "model-bw_tkdi.json", "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(self.directory + "model-bw_tkdi.h5")

        self.json_file_smn = open(self.directory + "model-bw_smn.json", "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(self.directory + "model-bw_smn.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("ASL Recognition using CNN")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1450x770")
        self.root.config(bg='powderblue')
        self.panel = tk.Label(self.root)
        self.panel.place(x=80, y=50, width=640, height=640)
        self.panel.config(bg='powderblue') # panel showing original image captured

        self.panel_pre_label_heading = tk.Label(self.root)
        self.panel_pre_label_heading.place(x=770, y=95)
        self.panel_pre_label_heading.config(text="IMAGE PREPROCESSING STEPS", font=('arial 30 bold'), bg='powderblue', fg='steelblue')

        self.panel_pre1=tk.Label(self.root)
        self.panel_pre1.place(x=770, y=145, width=310, height=310)
        self.panel_pre_label1 = tk.Label(self.root)
        self.panel_pre_label1.place(x=865, y=405)
        self.panel_pre_label1.config(text="1. Grayscale", font=("Times", 15, "bold"), bg='powderblue')

        self.panel_pre2 = tk.Label(self.root)
        self.panel_pre2.place(x=1090, y=145, width=310, height=310)
        self.panel_pre_label2 = tk.Label(self.root)
        self.panel_pre_label2.place(x=1180, y=405)
        self.panel_pre_label2.config(text="2. Gaussian Blur", font=("Times", 15, "bold"), bg='powderblue')

        self.panel_pre3 = tk.Label(self.root)
        self.panel_pre3.place(x=770, y=465, width=310, height=310)
        self.panel_pre_label3 = tk.Label(self.root)
        self.panel_pre_label3.place(x=820, y=720)
        self.panel_pre_label3.config(text="3. Adaptive Thresholding", font=("Times", 15, "bold"), bg='powderblue')
        # self.panel_pre4 = tk.Label(self.root)
        # self.panel_pre4.place(x=740, y=345, width=250, height=250)
        # self.panel_pre_label4 = tk.Label(self.root)
        # self.panel_pre_label4.place(x=10, y=345)
        # self.panel_pre_label4.config(text="Current Character :", font=("Times", 40, "bold"), bg='powderblue')

        self.panel2 = tk.Label(self.root)  # initialize image panel which shows final preprocessed image
        self.panel2.place(x=1090, y=465, width=310, height=310)
        self.panel_pre_label5 = tk.Label(self.root)
        self.panel_pre_label5.place(x=1180, y=720)
        self.panel_pre_label5.config(text="4. Binary Inversion", font=("Times", 15, "bold"), bg='powderblue')



        self.T = tk.Label(self.root)
        self.T.place(x=250, y=15)
        self.T.config(text="ASL Recognition using CNN", font=('arial 45 bold'), fg='steelblue', bg='powderblue')
        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Current Character :", font=("Times", 40, "bold") ,bg='powderblue')
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2image = cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 2)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            #displaying greyscale image in panel_pre1
            self.current_image = Image.fromarray(gray)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel_pre1.imgtk = imgtk
            self.panel_pre1.config(image=imgtk)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            # displaying gaussianBlur image in panel_pre2
            self.current_image = Image.fromarray(blur)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel_pre2.imgtk = imgtk
            self.panel_pre2.config(image=imgtk)

            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # displaying adaptive threshold image in panel_pre3
            self.current_image = Image.fromarray(th3)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel_pre3.imgtk = imgtk
            self.panel_pre3.config(image=imgtk)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            #displaying image after binary inversion in panel2
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.predict(res)
            self.panel3.config(text=self.current_symbol, font=("Courier", 50) ,bg='powderblue')
        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (310, 310))
        result = self.loaded_model.predict(test_image.reshape(1, 310, 310, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 310, 310, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 310, 310, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 310, 310, 1))
        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        # LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        if (self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
            prediction1 = {}
            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if (prediction1[0][0] == 'S'):
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self):

        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")




print("Starting Application...")
pba = Application()
pba.root.mainloop()
