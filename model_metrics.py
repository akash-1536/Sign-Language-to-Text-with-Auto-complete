import os
import random
import cv2
from keras.models import model_from_json
import operator
from string import ascii_uppercase
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Y_test=[]
Y_Pred1=[]
Y_Pred2=[]

class Test:

    def __init__(self):
        self.directory = 'model/'
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
        self.calc_metrics()


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
        current_prediction = prediction[0][0]

        # LAYER 2
        if (current_prediction == 'D' or current_prediction == 'R' or current_prediction == 'U'):
            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            current_prediction = prediction[0][0]

        if (current_prediction == 'D' or current_prediction == 'I' or current_prediction == 'K' or current_prediction == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            current_prediction = prediction[0][0]

        if (current_prediction == 'M' or current_prediction == 'N' or current_prediction == 'S'):
            prediction = {}
            prediction['M'] = result_smn[0][0]
            prediction['N'] = result_smn[0][1]
            prediction['S'] = result_smn[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            current_prediction = prediction[0][0]

        return current_prediction



    def calc_metrics(self):
        fin_path = "dataset_massey_univ/data/preprocessed/test"
        dir_list = os.listdir(fin_path)


        for letter in dir_list:
            if letter=='0':
                continue
            path1 = fin_path + "/" + letter
            dir_list1 = os.listdir((path1))
            for img in dir_list1:
                prob = random.uniform(0, 1)
                if (prob >= 0.95):
                    Y_test.append(letter)
                    path2 = path1 + "/" + img
                    pre_img = cv2.imread(path2)

                    minValue = 70
                    img_height = pre_img.shape[0]
                    img_width = pre_img.shape[1]
                    top = 0
                    bottom = 0
                    left = 0
                    right = 0
                    if img_height > img_width:
                        left = (img_height - img_width) // 2
                        right = img_height - img_width - left
                    elif img_height < img_width:
                        top = (img_width - img_height) // 2
                        bottom = img_width - img_height - top
                    resized1 = cv2.copyMakeBorder(pre_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                  value=[0, 0, 0])
                    dim = (310,310)
                    resized2 = cv2.resize(resized1, dim, interpolation=cv2.INTER_AREA)
                    if (letter == '0'):
                        dim = (310,310)
                        resized2 = cv2.resize(pre_img, dim, interpolation=cv2.INTER_AREA)

                    gray = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 2)
                    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    pred_char2 = self.predict(res)
                    Y_Pred2.append(pred_char2)

        corr = 0
        for i in range(0, len(Y_test)):
            if Y_test[i] == Y_Pred2[i]:
                corr = corr + 1

        print("\n\nMETRICS:")
        print("\nAccuracy= ", accuracy_score(Y_test, Y_Pred2))

        print("\n\nCLASSIFICATION REPORT:")
        print(classification_report(Y_test, Y_Pred2))

        cm = confusion_matrix(Y_test, Y_Pred2)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='0.0f')
        plt.xticks(ticks=np.arange(26 + 0.5),
                   labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        plt.yticks(ticks=np.arange(26 + 0.5),
                   labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.title("Confusion matrix")
        plt.show()


pba = Test()


