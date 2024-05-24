import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import os


class ImageApp:
    def __init__(self):
        self.model = load_model('C:/Users/DELL/Downloads/PFE_model_save.h5')

    def import_image(self, image_path):
        if image_path:
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            st.image(self.image, channels="BGR")

    def extract_image(self):
        command = f"yolo task=detect mode=predict model=C:/Users/DELL/Downloads/Yolov8/best.pt conf=0.80 source={self.image_path} save=true save_txt=true save_crop=true"
        os.system(command)

        output_base_dir = "C:/streamlit/module1/runs/detect"
        predict_dirs = [d for d in os.listdir(output_base_dir) if d.startswith('predict')]
        if not predict_dirs:
            st.error("Erreur : aucun répertoire 'predict' trouvé.")
            return

        latest_predict_dir = sorted(predict_dirs, key=lambda x: int(x.replace('predict', '') or '0'))[-1]
        crop_dir = os.path.join(output_base_dir, latest_predict_dir, "crops", "licence-plate")

        if not os.path.exists(crop_dir):
            st.error("Erreur : le répertoire des cultures n'existe pas.")
            return

        extracted_images = [img for img in os.listdir(crop_dir) if img.endswith('.jpg')]

        if extracted_images:
            self.extracted_image_path = os.path.join(crop_dir, extracted_images[0])  # Prendre le premier fichier trouvé
            self.plate = cv2.imread(self.extracted_image_path)
            st.image(self.plate, channels="BGR")
        else:
            st.error("Erreur : l'image extraite n'a pas été trouvée.")

    def start_processing(self):
        char = self.segment_characters(self.plate)

        for i in range(min(11, len(char))):
            st.image(char[i], channels="GRAY")

        plate_number, predicted_chars = self.show_results(char)
        st.write(f"Numéro de plaque : {plate_number}")

        st.write("Predicted Characters:")
        for i, ch in enumerate(char):
            st.image(ch, caption=f'Predicted: {predicted_chars[i]}', channels="GRAY")

    def find_contours(self, dimensions, img):
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lower_width, upper_width, lower_height, upper_height = dimensions

        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        x_cntr_list = []
        img_res = []
        for cntr in cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
                x_cntr_list.append(intX)
                char_copy = np.zeros((44, 24))
                char = img[intY:intY + intHeight, intX:intX + intWidth]
                char = cv2.resize(char, (20, 40))
                char = cv2.subtract(255, char)
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0
                img_res.append(char_copy)

        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = [img_res[idx] for idx in indices]
        img_res = np.array(img_res_copy)
        return img_res

    def segment_characters(self, image):
        img_lp = cv2.resize(image, (333, 75))
        if len(img_lp.shape) == 3:
            img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        else:
            img_gray_lp = img_lp
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]
        img_binary_lp[0:3, :] = 255
        img_binary_lp[:, 0:3] = 255
        img_binary_lp[72:75, :] = 255
        img_binary_lp[:, 330:333] = 255

        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
        char_list = self.find_contours(dimensions, img_binary_lp)
        return char_list

    def fix_dimension(self, img):
        new_img = np.zeros((28, 28, 3))
        for i in range(3):
            new_img[:, :, i] = img
        return new_img

    def show_results(self, char):
        dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        output = []
        predicted_chars = []
        for i, ch in enumerate(char):
            img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            img = self.fix_dimension(img_)
            img = img.reshape(1, 28, 28, 3)
            probabilities = self.model.predict(img)[0]
            y_ = np.argmax(probabilities)
            character = dic[y_]
            output.append(character)
            predicted_chars.append(character)
        plate_number = ''.join(output)
        return plate_number, predicted_chars

def main():
    st.title("Image Importer")
    st.sidebar.title("Options")

    image_app = ImageApp()

    uploaded_image = st.sidebar.file_uploader("Importer une image", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if uploaded_image:
        image_app.import_image(uploaded_image.name)

    extract_button = st.sidebar.button("Extraire")
    if extract_button:
        image_app.extract_image()

    start_button = st.sidebar.button("Démarrer")
    if start_button:
        image_app.start_processing()

if __name__ == "__main__":
    main()
