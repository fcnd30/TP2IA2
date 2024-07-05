import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image
import io
import os
import uvicorn
from fastapi import FastAPI
app = FastAPI()

# Charger les signatures des visages
signatures_class = np.load('FaceSignature_db.npy')
X = signatures_class[:, 0: -1].astype('float')
Y = signatures_class[:, -1]

def main():
    st.title("Reconnaissance faciale et affichage des images similaires")
    image = st.file_uploader("Choisir une image", type=['png', 'jpeg', 'jpg'])
    
    if image is not None:
        # Convertir l'image uploadée pour traitement avec OpenCV
        image_stream = io.BytesIO(image.getvalue())
        image = Image.open(image_stream)
        img = np.array(image)

        img_resize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        # Afficher immédiatement l'image uploadée
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Trouver les emplacements des visages dans l'image uploadée
        faces_current = face_recognition.face_locations(img_resize)
        encodes_current = face_recognition.face_encodings(img_resize, faces_current)

        # Parcourir toutes les images PNG dans le dossier spécifié
        images_path = './images'
        for filename in os.listdir(images_path):
            if filename.endswith('.png'):
                path = os.path.join(images_path, filename)
                image_to_check = Image.open(path)
                image_to_check_array = np.array(image_to_check)
                
                # Convertir et redimensionner comme l'image uploadée pour comparaison
                image_to_check_array_resized = cv2.resize(image_to_check_array, (0, 0), None, 0.25, 0.25)
                image_to_check_array_resized = cv2.cvtColor(image_to_check_array_resized, cv2.COLOR_BGR2RGB)
                
                # Trouver les emplacements des visages et les encodages dans l'image du dossier
                faces_to_check = face_recognition.face_locations(image_to_check_array_resized)
                encodes_to_check = face_recognition.face_encodings(image_to_check_array_resized, faces_to_check)
                
                for encode_check in encodes_to_check:
                    matches = face_recognition.compare_faces(encodes_current, encode_check)
                    if True in matches:
                        # Afficher l'image si une similarité est trouvée
                        st.image(image_to_check, caption=f"Similaire: {filename}")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8801)
    main()


