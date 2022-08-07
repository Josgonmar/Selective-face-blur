import cv2, streamlit
import numpy as np
import io, base64
from PIL import Image

class ModelParams():
    mean = [104, 117, 123]
    scale = 1.0
    in_width = 300
    in_height = 300

class FaceBlurApp:
    __face_detector = None
    __model_params = None
    __max_blur_factor = 5
    __min_blur_factor = 1

    def __init__(self):
        self.__loadModel()
        streamlit.title('Selective face blurrer')

    def run(self):
        uploaded_image = streamlit.file_uploader('Upload an image:', type=['png', 'jpg'])

        if uploaded_image is not None:
            image = self.__toOpenCV(uploaded_image)
            input_col, output_col = streamlit.columns(2)

            blob = cv2.dnn.blobFromImage(image, scalefactor=self.__model_params.scale, size=(self.__model_params.in_width, self.__model_params.in_height),
                                            mean=self.__model_params.mean, swapRB=False, crop=False)
            self.__face_detector.setInput(blob)

            detections = self.__face_detector.forward()
            detection_threshold = streamlit.slider('Detection threshold', 0.5, 1.0, 0.75)
            blur_factor = streamlit.slider('Blur factor', self.__min_blur_factor, self.__max_blur_factor, self.__min_blur_factor)

            input_image, faces = self.__generateFaceBlurBounds(image, detections, detection_threshold)

            with input_col:
                streamlit.subheader('Original input image')
                streamlit.image(input_image, channels='BGR', use_column_width=True)

            selected_faces = self.__generateSelections(faces)
            output_image = self.__blurFaces(image, selected_faces, blur_factor)
            
            with output_col:
                streamlit.subheader('Final image')
                streamlit.image(output_image, channels='BGR', use_column_width=True)

            streamlit.markdown(self.__getDownloadLink(Image.fromarray(output_image[:,:,::-1]), 'JPEG', 'output.jpeg'),
                unsafe_allow_html=True)
    
    def __generateSelections(self, faces):
        arg = {}
        selected_faces = []
        for count in range(len(faces)): arg['Face ' + str(count)] = count
        selections = streamlit.multiselect('Select which face(s) you would like to blur', arg)

        for data in faces:
            for face_num in selections:
                if data[0] == face_num:
                    selected_faces.append(data)

        return selected_faces

    def __getDownloadLink(self, img, format, filename):
        buffered = io.BytesIO()
        img.save(buffered, format = format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">Download output</a>'
        return href

    def __generateFaceBlurBounds(self, image, detections, threshold):
        height, width = image.shape[:2]
        input_image = image.copy()
        face_count = 0
        detected_faces_data = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                txt_coords = self.__getTextPosition(str(face_count), x1, y1, x2, y2)
                cv2.putText(input_image, str(face_count), txt_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, 2)
                cv2.rectangle(input_image, (x1,y1), (x2,y2), (0,0,255), 1, 1)

                detected_faces_data.append(['Face ' + str(face_count), x1, y1, x2, y2])
                face_count += 1
        
        return input_image, detected_faces_data
    
    def __blurFaces(self, image, faces, factor):
        output_image = image.copy()
        blurred_image = image.copy()
        ellipse_mask = np.zeros(image.shape, dtype=image.dtype)
        for data in faces:
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[3])
            y2 = int(data[4])

            face = output_image[y1:y2, x1:x2, :]
            face = self.__blurFace(face, factor)

            blurred_image[y1:y2, x1:x2, :] = face

            e_center = ((x2-x1)/2 + x1, (y2-y1)/2 + y1)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0

            ellipse_mask = cv2.ellipse(ellipse_mask, (e_center, e_size, e_angle), (255, 255, 255), -1, cv2.LINE_AA)
            np.putmask(output_image, ellipse_mask, blurred_image)

        return output_image
    
    def __blurFace(self, face, factor):
        height, width  = face.shape[:2]
        
        w_k = int(width/((self.__max_blur_factor + 1) - factor))
        h_k = int(height/((self.__max_blur_factor + 1) - factor))
        if w_k%2 == 0: w_k += 1
        if h_k%2 == 0: h_k += 1

        if face.shape[0] != 0 and face.shape[1] != 0:
            blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
        else:
            blurred = face

        return blurred

    def __loadModel(self):
        self.__face_detector = cv2.dnn.readNetFromCaffe('../../model/deploy.prototxt', '../../model/res10_300x300_ssd_iter_140000.caffemodel')
        self.__model_params = ModelParams()
    
    def __toOpenCV(self, src):
        raw_bytes = np.asarray(bytearray(src.read()), dtype=np.uint8)
        dst = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        return dst
    
    def __getTextPosition(self, text, x1, y1, x2, y2):
        txt_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        px = (x2-x1)/2 + x1 - txt_size[0]/2
        py = (y2-y1)/2 + y1 + txt_size[1]/2

        return np.array([px, py], dtype=int)

if __name__ == "__main__":
    FaceBlurrerApp_obj = FaceBlurApp()
    FaceBlurrerApp_obj.run()