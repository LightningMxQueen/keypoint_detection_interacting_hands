import transforms as T

def get_transform():
    transforms =[]
    transforms.append(T.ToTensor())
    return T.Compose(transforms)