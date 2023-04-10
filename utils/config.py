class Config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update_from_dict(self, d):
        self.__dict__.update(d)