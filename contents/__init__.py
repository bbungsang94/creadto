

class Contents:
    def __init__(self):
        losses = {
        }
        optimizers = {
        }

        self.motherboard = {
            'loss': losses,
            'optimizer': optimizers
        }

    def __getitem__(self, key):
        result = None
        module = key.split('_')
        if module[0] in self.motherboard[module[-1]]:
            result = self.motherboard[module[-1]][module[0]]
        return result
