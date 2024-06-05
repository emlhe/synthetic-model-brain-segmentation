import torchio as tio

def preprocess(ensure_shape_mul = 32, num_classes=41):
    return tio.Compose([
                tio.EnsureShapeMultiple(ensure_shape_mul, method='crop'),  # for the U-Net : doit être un multiple de 2**nombre de couches
                tio.OneHot(num_classes=num_classes),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                ])
     
def preprocess_label_map():
    # remapping = {4: 3, 5: 1, 6: 2, 7: 2, 8:2, 9:2, 10:2, 11:2, 12:3, 13:4, 14: 2, 15: 2, 16: 5, 17:2,18:2,21:1,22:2,23:3,24:3,25:1,26:2,27:2,28:2,29:2,30:2,31:2,32:2,33:2,34:2,37:6,38:3}
    remapping = {4: 1, 5: 1, 6: 1, 7: 1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14: 1, 15: 1, 16: 1, 17:1,18:1,21:2,22:2,23:2,24:2,25:2,26:2,27:2,28:2,29:2,30:2,31:2,32:2,33:2,34:2,37:3,38:2}
    return tio.Compose([
        tio.RemapLabels(remapping),
        tio.RemoveLabels([19,20,35,36,39,40]),
        tio.SequentialLabels()
    ])
        
def augment():
    return tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=(0.8,1.0),degrees=(0,3), translation=(0,10)): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                    },
                    p=0.75,
                ),
                tio.RandomMotion(p=0.2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),
                tio.RandomFlip(axes=('LR',), flip_probability=0.2),
                tio.RandomBiasField(coefficients = 0.3, p=0.5),
                tio.RandomNoise(std=0.1, p=0.25),
                                ])