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
def compute_face_embedding(img_path):
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

'''
    Function used to compute the cosine similarity between
    to embeddings
    
    returns -1 if they are opposite
    returns 1 if they are identical
'''
def compute_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1[0], embedding2[0])
    
    # compute the magnitude of each embedding
    norm_embedding1 = np.linalg.norm(embedding1[0])
    norm_embedding2 = np.linalg.norm(embedding2[0])
    
    # compute the cosine similarity
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    
    return similarity

if __name__ == '__main__':
    image_paths_testing   = get_image_paths("database_testing_photos/")
    image_paths_reference = get_image_paths("database_reference_photos/")
    reference_embeddings  = []
    test_embeddings       = []
    
    # compute the reference embeddings
    for img in image_paths_reference:
        embedding = compute_face_embedding(img)
        
        # append the embedding to the list
        reference_embeddings.append([embedding,img])
        
    # go through the test images and compute the embeddings
    for img in image_paths_testing:
        embedding = compute_face_embedding(img)
        
        # append the embedding to the list
        test_embeddings.append([embedding, img])
        
        
    # go through the test_embeddings and find the closest refernce_embedding
    for embedding in test_embeddings:
        max_similarity = [-1, ""]
        
        for ref_embedding in reference_embeddings:
            similarity = compute_similarity(embedding[0], ref_embedding[0])
            similarity_max = max_similarity[0]
            
            if (similarity > float(max_similarity[0])):
                max_similarity = [similarity, ref_embedding[1]]
        
        
        print("The image with max similarity to {0} is {1}".format(embedding[1], max_similarity[1]))
     
    
    