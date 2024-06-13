import torchio as tio

def preprocess(num_classes):
    return tio.Compose([
                #tio.EnsureShapeMultiple((shape,shape,shape), method='crop'), Pas besoin quand patch # for the U-Net : doit être un multiple de 2**nombre de couches
                tio.OneHot(num_classes=num_classes),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.ToCanonical(),
                tio.Resample(1)
                ])
     
def preprocess_label_map():
    #remapping = {2:3, 3:2, 4:1, 3:3, 4:3, 5:1, 6:1, 7:4, 8:4, 9:4, 10:4, 11:2, 12:3, 14:2, 15:2, 17:2, 18:2, 21:1, 22:2, 23:3, 24:3, 25:1, 26:2, 27:4, 28:4, 29:4, 30:4, 31:2, 32:2, 33:2, 34:2, 37:5, 38:3}
    #remapping = {4: 1, 5: 1, 6: 1, 7: 1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14: 1, 15: 1, 16: 1, 17:1,18:1,21:2,22:2,23:2,24:2,25:2,26:2,27:2,28:2,29:2,30:2,31:2,32:2,33:2,34:2,37:3,38:2}
    remapping = {2:3,3:2,4:1,5:1,7:3,8:2,10:4,11:4,12:4,13:4,14:1,15:1,16:3,17:2,18:2,24:1,26:2,28:2,29:0,30:0,41:3,42:2,43:1,44:1,46:3,47:2,49:4,50:4,51:4,52:4,53:2,54:2,58:2,60:2,61:0,62:0,64:5,72:0,77:3,85:0}    
    return tio.Compose([
        tio.RemapLabels(remapping),
        tio.SequentialLabels()
    ])
        
def augment():
    return tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=(0.9,1.1),degrees=20, translation=(-10,10)): 0.8,
                    tio.RandomElasticDeformation(num_control_points = 12, max_displacement = 8): 0.2,
                    },
                    p=0.75,
                ),
                tio.RandomMotion(p=0.2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),
                tio.RandomFlip(axes=('LR',), flip_probability=0.2),
                tio.RandomBiasField(coefficients = 0.5, order = 3, p=0.5),
                tio.RandomNoise(mean = 0, std=(0.005, 0.1), p=0.25),
                                ])