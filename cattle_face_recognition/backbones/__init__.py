from .new_gam_iresnet import new_gam_iresnet100

def get_model(name, **kwargs):

    if name == 'new_gam_r100':
        return new_gam_iresnet100(False,**kwargs)
    else:
        raise ValueError()
