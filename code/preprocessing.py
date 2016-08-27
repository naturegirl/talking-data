# shared preprocessing functions

from sklearn.preprocessing import OneHotEncoder

def encode_features(features):
    """
    Use one hot encoding to encode the 2 categorical features
    phone_brand_id and device_model_id
    """
    enc = OneHotEncoder()
    enc.fit(features)
    return enc.transform(features).toarray()
