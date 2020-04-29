from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DistractedDriverTestDataset
from parameters import test_dir


def test_model(model_class, state_dict, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    model = model_class()
    model.load_state_dict(state_dict)
    model = model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
            normalize
        ])

    test_data = DistractedDriverTestDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    output_data = []

    model.eval()
    with torch.no_grad():
        for i, (img_names, images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            logits = model(images)
            pred = F.softmax(logits, dim=1)
            pred = pred.tolist()

            for name, score in zip(img_names, pred):
                output_row = [name] + score
                output_data.append(output_row)
            if i == 50:
                break
    output_df = pd.DataFrame(data=output_data,
                             columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7','c8', 'c9'])
    output_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    from knowledge_distillation import Student_network
    state_dict = torch.load('models/student_04-26_21-35_val.pt')
    test_model(model_class=Student_network, state_dict=state_dict, save_path='submissions.csv')