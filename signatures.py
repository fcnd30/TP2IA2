import cv2, numpy as np, face_recognition, os


#imageDb path
path = './images'

# Global variables
image_list = [] # List of images
name_list = [] # List of image names

#Gral all images from the folder
myList = os.listdir(path)

#print(myList)

#Load images
for img in myList:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.jpeg']:
        curIimg = cv2.imread(os.path.join(path, img))
        image_list.append(curIimg)
        imgName = os.path.splitext(img)[0]
        name_list.append(imgName)

#Define a function to detect and extract features therefrom    
def findEncodings(img_list, ImgName_list):
    """Définit une fonction pour détecter les visages et extraire leurs caractéristiques.

    Args:
        img_list (list): Liste d'images en format BGR.
        ImgName_list (list): Liste des noms d'images.
    """
    signatures_db = []
    count = 1
    for myImg, name in zip(img_list, ImgName_list):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Vérifie si des encodages sont trouvés
            signature = encodings[0]
            signature_class = signature.tolist() + [name]
            signatures_db.append(signature_class)
            print(f'{int((count/ len(img_list))*100)} % extracted')
        else:
            print(f"No face found in image {name}. Skipping...")
        count += 1
    signatures_db = np.array(signatures_db)   
    np.save('FaceSignature_db.npy', signatures_db)
    print('Signature_db stored')

def main():
    findEncodings(image_list, name_list)   

if __name__ == '__main__':
    main() 
