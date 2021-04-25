import torch

class DEC(torch.nn.Module):
    def __init__(self, init_np_centers, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.centers = torch.nn.Parameter(torch.tensor(init_np_centers), requires_grad=True)

    def prediction(self, data):
        return dec_prediction(self.centers, data, self.alpha)

    def prediction_hard(self, data):
        return self.prediction(data).argmax(1)

    def prediction_hard_np(self, data):
        return self.prediction_hard(data).cpu().numpy()

    def loss_dec_compression(self, data_batch):
        prediction = dec_prediction(self.centers, data_batch, self.alpha)
        loss = dec_compression_loss_fn(prediction)
        return loss


def dec_prediction(centers, data, alpha=1.0):
    ta = centers.unsqueeze(0)
    tb = data.unsqueeze(1)
    squared_diffs = (ta - tb).pow(2).sum(2)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob

def dec_compression_value(pred_labels):
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p

def dec_compression_loss_fn(p):
    q = dec_compression_value(p).detach().data
    loss = -1.0 * torch.mean(torch.sum(q * torch.log(p + 1e-8), dim=1))
    return loss
