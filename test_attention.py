import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DistractedDriverTestDataset
from parameters import test_dir


def test_model(model, state_dict, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
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
            output = model(images)
            logits = output['logits']
            att = output['att']

            pred = F.softmax(logits, dim=1)
            pred = pred.tolist()

            for idx in range(len(img_names)):
                name = img_names[idx]
                score = pred[idx]
                current_att = att[idx][0].cpu().numpy()

                output_row = [name] + score
                output_data.append(output_row)
                fig, ax = plt.subplots(2)
                cb = ax[1].imshow(current_att, cmap='gray')
                ax[0].imshow(Image.open(os.path.join(test_dir, name)))
                # plt.colorbar()
                fig.suptitle(f'{np.round(score, 3)}')
                fig.colorbar(cb, ax=ax[1])
                plt.show()
                plt.close()

    output_df = pd.DataFrame(data=output_data,
                             columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7','c8', 'c9'])
    output_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    import os
    from attention_model import ResNetAttention
    model = 'att_lstm_05-03_00-25_val.pt'
    submission_path = os.path.join('submissions', model[:-3] + '.csv')
    state_dict = torch.load(os.path.join('models', model))
    test_model(model=ResNetAttention, state_dict=state_dict, save_path=submission_path)
