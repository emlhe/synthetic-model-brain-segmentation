import torchio as tio

def get_subjects(image_paths, label_paths=None): 
    subjects = []

    if label_paths == None :
        for image_path in image_paths:
            subject = MySubject(
                img=tio.ScalarImage(image_path),
                subject=str(image_path).split("/")[-1].split(".nii.gz")[0]
            )
            subjects.append(subject)
    else:
        for (image_path, label_path) in zip(image_paths, label_paths):

            subject = MySubject(
                img=tio.ScalarImage(image_path),
                seg=tio.LabelMap(label_path),
                subject=str(image_path).split("/")[-1].split(".nii.gz")[0]
            )
            subjects.append(subject)
    
    return subjects


class MySubject(tio.Subject):
    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-5
        kwargs['absolute_tolerance'] = 1e-5
        return super().check_consistent_attribute(*args, **kwargs)