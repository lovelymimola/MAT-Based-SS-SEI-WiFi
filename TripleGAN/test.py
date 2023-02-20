from torch.utils.data import DataLoader, TensorDataset
from get_dataset_5label import *
from torch.nn import functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def test(netA, netC, test_dataloader):
    netA.eval()
    netC.eval()
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
            '''
            自编码器
            输入：测试集数据data
            输出：测试集数据的潜在特征features = output_of_netA[0]
                 测试集数据的重构数据r_data = output_of_netA[1]
            '''
            output_of_netA = netA(data)
            features = output_of_netA[0]

            '''
            分类器
            输入：测试集数据的潜在特征features = output_of_netA[0]
            输出：测试集数据的logits = output_of_netC
            '''
            output_of_netC = netC(features)
            netC_value = F.log_softmax(output_of_netC, dim=1)
            pred = netC_value.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target)-1] = pred.tolist()
            target_real[len(target_real):len(target)-1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

    # # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("TripleGAN_15label/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()
    #
    # # 将原始标签存下来
    #
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("TripleGAN_15label/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
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
    netA = torch.load("model_weight/netA_n_classes_10_label5_unlabel95_rand30.pth")
    netC = torch.load("model_weight/netC_n_classes_10_label5_unlabel95_rand30.pth")
    test(netA, netC, test_dataloader)

if __name__ == '__main__':
   main()
