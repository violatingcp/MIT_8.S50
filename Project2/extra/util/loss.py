import torch

class MSE():
    def __init__(self,ntargets):
        self.ntargets = ntargets
    def __call__(self,preds,targets, weights=None):
        if self.ntargets == 1:
            preds = torch.sigmoid(preds).view(-1,1)
            targets = targets.view(-1,1)
        else:
            preds = torch.nn.functional.softmax(preds, dim=-1)
            targets = torch.nn.functional.one_hot(targets.long(), self.ntargets)
        weights = weights if weights is not None else torch.ones(preds.shape[0])
        weights = weights.view(-1,1)
        loss = (weights*(preds-targets)**2).mean()
        return loss
    def __repr__(self):
        return f"MSE ntargets{self.ntargets}"
    
class DiscoLoss():
    def __init__(self,background_only=True,background_label=1,power=1):
        self.backonly = background_only
        self.background_label = background_label
        self.power = power
    def __call__(self,pred,target,x_biased,weights=None):
        """
        Calculate the total loss (flat and MSE.)


        Parameters
        ----------
        pred : Tensor
            Tensor of predictions.
        target : Tensor
            Tensor of tar get labels.
        x_biased : Tensor
            Tensor of biased feature.
        """
        if self.backonly:
            mask = target==self.background_label
            x_biased = x_biased[mask]
            pred = pred[mask]
            target = target[mask]
            if weights is not None:
                weights =  weights[mask]
            else:
                weights = torch.ones_like(target)
            del mask
        disco = distance_corr(x_biased,pred,normedweight=weights,power=self.power)
        return disco 
    def __repr__(self):
        str1 = "DisCo Loss: background_label={}, power={}".format(self.background_label,self.power)
        return "\n".join([str1])

def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    
    
    xx = var_1.view(-1, 1).expand(len(var_1), len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.expand(len(var_1),len(var_1)).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()
    del xx,yy

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.expand(len(var_1),len(var_1)).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).expand(len(var_1), len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)
    del amat 

    xx = var_2.view(-1, 1).expand(len(var_2), len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.expand(len(var_2),len(var_2)).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()
    del xx,yy

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.expand(len(var_2),len(var_2)).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).expand(len(var_2), len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)
    del bmat 

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)
    del Bmat, Amat
    
    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr

