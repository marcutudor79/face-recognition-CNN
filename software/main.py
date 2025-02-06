from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import numpy as np
from operator import itemgetter

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
    # print(similarity)
    return similarity

if __name__ == '__main__':
    image_paths_testing   = get_image_paths("database_testing_photos/")
    image_paths_reference = get_image_paths("database_reference_photos/")
    ref_emb_and_img_paths = []
    tst_emb_and_img_paths = []

    # compute the reference embeddings
    for img in image_paths_reference:
        embedding = compute_face_embedding(img)

        # append the embedding to the list
        ref_emb_and_img_paths.append([embedding,img])

    # go through the test images and compute the embeddings
    for img in image_paths_testing:
        embedding = compute_face_embedding(img)

        # append the embedding to the list
        tst_emb_and_img_paths.append([embedding, img])


    # go through the test_embeddings and find the closest reference_embedding
    for tst_emb_img_path in tst_emb_and_img_paths:
        similarity_and_img_paths = []

        for ref_emb_img_path in ref_emb_and_img_paths:
            similarity = compute_similarity(tst_emb_img_path[0], ref_emb_img_path[0])
            similarity_and_img_paths.append([similarity, ref_emb_img_path[1]])

        # sort the array based on the similarity
        similarity_and_img_paths = sorted(similarity_and_img_paths, key=itemgetter(0))

        print("The image with maximal similarity to {0} is {1}".format(tst_emb_img_path[1], similarity_and_img_paths[-1][1]))
        print("The image with second similiraty to {0} is {1}".format(tst_emb_img_path[1], similarity_and_img_paths[-2][1]))
        print("The image with third similarity to {0} is {1}".format(tst_emb_img_path[1], similarity_and_img_paths[-3][1]))
        print("------------------------------------------------------------------------------------------------------------")
