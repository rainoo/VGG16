import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import Vgg16

def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset= test_data,
                                   batch_size= 1,
                                   shuffle= True,
                                   num_workers= 8)

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for (test_data_x, test_data_y) in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)
            test_corrects += (pre_label == test_data_y.data)
            test_num += test_data_x.size(0)
        test_acc = test_corrects.double().item() / test_num
        print(f"测试的准确率为: {test_acc}")

if __name__ == '__main__':
    model = Vgg16()
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    #
    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    #
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #         model.eval()
    #         output = model(b_x)
    #         pre_label = torch.argmax(output, dim=1)
    #         result = pre_label.item()
    #         label = b_y.item()
    #         print(f'预测结果是{classes[result]}, 真实标签是{classes[label]}')
