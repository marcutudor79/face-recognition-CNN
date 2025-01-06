from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import numpy as np

# Crete face detection pipeline, keep it on CPU 
mtcnn = MTCNN(keep_all = True, device = 'cpu')

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

'''
    Function to compute the face weights
    based on the img_path
'''
def compute_face_weight(img_path):
    img = Image.open(img_path)
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn(img)
        embeddings = resnet(faces.to('cpu')).detach().cpu().numpy()
        return embeddings
    else:
        print(f"No face found in {img_path}")
        return None

'''
    Function used to automatically retrieve the names of 
    paths to the photos in the database
'''
def get_image_paths(database_path):
    image_paths = []
    
    for root, dirs, files in os.walk(database_path):
        for name in files:
            image_paths.append(os.path.join(root, name))
            
    return image_paths
    
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

if __name__ == '__main__':
    image_paths_testing = get_image_paths("database_testing_photos/")
    image_paths_reference = get_image_paths("database_reference_photos/")
    
    reference_embeddings = [(compute_face_weight(img_path), img_path) for img_path in image_paths_reference]
    
    for test_img in image_paths_testing:
        test_embedding = compute_face_weight(test_img)
        
        if test_embedding is not None:
            distances = [(calculate_distance(test_embedding, ref_embedding), ref_img_path) for ref_embedding, ref_img_path in reference_embeddings]
            distances.sort(key = lambda x: x[0])
            closest_images = distances[:3]
    
            print(f"\nClosest images to {test_img}:")
            for dist, img_path in closest_images:
                print(f"Distance: {dist:.3f}, Reference Image Path: {img_path}")
    
    