import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_5label import *
from model_complexcnn_onlycnn import *
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def test(model, test_dataloader, rand_num):
    model.eval()
    correct = 0
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target)-1] = pred.tolist()
            target_real[len(target_real):len(target)-1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

    # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("CNN_n_classes_10_10label_rand"+str(rand_num)+"/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    # # 将原始标签存下来
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("CNN_n_classes_10_10label_rand"+str(rand_num)+"/Y_real.xlsx")
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
    rand_num = 30
    ft = 62
    X_test, Y_test = TestDataset(ft)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = torch.load("model_weight/CNN_n_classes_10_5label_95unlabel_rand"+str(rand_num)+".pth")
    print(model)
    test(model,test_dataloader,rand_num)

if __name__ == '__main__':
   main()
