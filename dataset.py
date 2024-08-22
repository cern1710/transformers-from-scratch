import torch
import torch.utils.data as data
import transformer
import torch.nn.functional as F

class ReverseDataset(data.Dataset):
    def __init__(self, num_classes: int, seq_len: int, size: int):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_classes,
                                  size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        inputs = self.data[index]
        labels = torch.flip(inputs, dims=(0,))
        return inputs, labels

class ReversePredictor(transformer.TransformerPredictor):
    def _calculate_loss(self, batch, mode='train'):
        inputs, labels = batch
        inputs = F.one_hot(inputs, num_classes=self.num_classes).float()

        preds = self.forward(inputs, include_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1,preds.size(-1)),
                               labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss, acc

    def training_step(self, batch):
        loss, _ = self._calculate_loss(batch, mode='train')
        return loss

    def validation_step(self, batch):
        _ = self._calculate_loss(batch, mode='val')

    def test_step(self, batch):
        _ = self._calculate_loss(batch, mode='test')