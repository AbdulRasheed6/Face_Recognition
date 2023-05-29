import keras
import numpy as np
from keras.models import Model
from encode.py  import image_to_encoding


model=keras.models.load_model('vgg_face.h5')


def findCosineSimilarity(source_representation, test_representation):
    a=  np.matmul(np.transpose(source_representation), test_representation)
    b= np.sum(np.multiply(source_representation, source_representation))
    c=np.sum(np.multiply(test_representation, test_representation))
    return 1- (a / (np.sqrt(b) * np.sqrt(c)))

vgg_face_descriptor=Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


database={}
# loop through the directory
for filename in os.listdir(inec_DIR):
    # the file path of each images
    f=os.path.join(inec_DIR, filename)
    
    
    database[filename.split('.')[0]]=image_to_encoding(f)
    

def who_is_it(image_path, database, model1, threshold):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model1 -- your custom vgg model instance in Keras
        threshold-  epsilon
    
    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding =  image_to_encoding(image_path)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute Cosine similarity distance between the target "encoding" and the current db_enc from the database. (≈ 3 line)
      
        db_enc1=model1.predict(db_enc)[0,:]
        encoding1=model1.predict(encoding)[0,:]
    
        cosine_similarity=findCosineSimilarity(db_enc1, encoding1)
        print('Cosine similarity: ', cosine_similarity)
    


        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if cosine_similarity < min_dist:
            min_dist = cosine_similarity
            identity = name
    ### END CODE HERE
    
    if min_dist > threshold:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

who_is_it('image',database, vgg_face_descriptor, 0.5 )    