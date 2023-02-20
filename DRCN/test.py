import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test(encoder, classifier, test_dataloader):
    encoder.eval()
    classifier.eval()
    correct = 0
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #data = data.unsqueeze(-1)
                target = target.to(device)
            features = encoder(data)
            output = F.log_softmax(classifier(features), dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

    # # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("DRCN_15label/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()
    #
    # # 将原始标签存下来
    #
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("DRCN_15label/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

def main():
    ft = 62
    X_test, Y_test = TestDataset(ft)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    encoder = torch.load('model_weight/encoder_complex_n_classes_10_label10_unlabel90_rand30.pth')
    classifier = torch.load('model_weight/classifier_complex_n_classes_10_label10_unlabel90_rand30.pth')
    test(encoder, classifier, test_dataloader)

if __name__ == '__main__':
   main()
