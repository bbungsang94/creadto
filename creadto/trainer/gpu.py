import torch

from creadto.trainer.base import Base
from creadto.utils.report import summary_device


class SingleGPURunner(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'SingleGPURunner'

    def _check_sanity(self):
        super()._check_sanity()
        self.report_state()

        self._model.to(self.device)
        if "to" in dir(self._loader):
            self._loader.to(self.device)
        if "to" in dir(self._evaluator):
            self._evaluator.to(self.device)

    def report_state(self):
        self.device = summary_device()

    def _write_log(self, **kwargs):
        properties = torch.cuda.get_device_properties(self.device)
        self._writer.add_scalars(kwargs['mode'] + '_' + properties.name + '_' + str(kwargs['epoch']),
                                 {'Memory(GB)': (properties.total_memory / (1024 ** 3)),
                                  'Loss(avg)': "%.4f" % kwargs['loss'],
                                  'Iter': kwargs['tick']})
        self._writer.flush()