"""Stage-3 affinity fine-tuning with frozen two-scale contact scorers."""

from .MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2


class MMDCGDTAModel_Stage3(MMDCGDTAModel_Stage2):
    def freeze_reconstructors(self):
        for reconstructor in self.reconstructors:
            for parameter in reconstructor.parameters():
                parameter.requires_grad = False

    def unfreeze_reconstructors(self):
        for reconstructor in self.reconstructors:
            for parameter in reconstructor.parameters():
                parameter.requires_grad = True
