import torch
import torch.nn as nn


class CEloss(nn.Module):
    def __init__(self, ig_idx):
        super(CEloss, self).__init__()

        self.ig_idx = ig_idx
        self.CEloss = torch.nn.CrossEntropyLoss(ignore_index= self.ig_idx)
        self.CEloss2 = torch.nn.CrossEntropyLoss()
        
    def forward(self, predict, predict2, target ,target2):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        
        loss = self.CEloss(predict.cuda(),target.long().cuda())
        loss_cls = self.CEloss2(predict2.cuda(),target2.float().cuda())
        return loss, loss_cls
