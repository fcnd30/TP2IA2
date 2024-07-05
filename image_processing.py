import os
import cv2
import numpy as np
import face_recognition

def extract_features(image_path):
    """Extraire les encodages des visages d'une image."""
    img = cv2.imread(image_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            return encodings[0]
        else:
            print(f"Aucun visage trouvé dans l'image à {image_path}.")
            return None
    else:
        print(f"Attention: L'image à {image_path} n'a pas pu être chargée.")
        return None

def process_images(root_folder):
    """
    Process each dataset folder within the root folder and extract face encodings,
    if their signature file does not already exist.
    """
    all_features = []  # List to store all features and metadata
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                features = extract_features(image_path)
                if features is not None:
                    # Append features, class name, and relative path to the list
                    class_name = os.path.basename(root)  # Assuming folder name is class name
                    all_features.append((features, class_name, image_path))
                    print(f'Processed: {file} -> Class: {class_name}')

    # Convert list to a NumPy array and save it
    signatures = np.array(all_features, dtype=object)
    np.save('FaceSignature_db.npy', signatures)
    print(f'Features successfully stored in FaceSignature_db.npy')


def save_image(img_upload, img_folder):
    if img_upload is not None:
        # Create a file path for the uploaded file in the specified folder
        img_path = os.path.join(img_folder, img_upload.name)
        # Open the file in binary write mode and write the image data
        with open(img_path, "wb") as f:
            f.write(img_upload.getbuffer())
        return img_path
    else:
        return None   

def main():
    # descriptors = ['glcm', 'Haralick', 'bitdesc', 'bitdesc_glcm', 'haralick_bitdesc']
    # for descriptor_type in descriptors:
    process_images('./images')

if __name__ == '__main__':
    main()

                
                

