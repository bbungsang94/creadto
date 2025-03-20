import torch

def wrapped_getattr(instance, name, default=None, wrapped_member_name='model'):
    ''' should be called from wrappers of model classes such as ClassifierFreeSampleModel'''

    if isinstance(instance, torch.nn.Module):
        # for descendants of nn.Module, name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # Otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(instance, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(instance, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(instance, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr        