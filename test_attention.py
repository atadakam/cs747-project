import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DistractedDriverTestDataset, CLASSES
from parameters import test_dir


def test_model(model, state_dict, save_path, visualize=False):
    LABELS = list(CLASSES.values())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10
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
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    output_data = []

    model.eval()
    with torch.no_grad():
        for i, (img_names, images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            print(images.shape)
            output = model(images)
            logits = output['logits']
            att = output['att']

            pred = F.softmax(logits, dim=1)
            pred = pred.tolist()

            for idx in range(len(img_names)):
                name = img_names[idx]
                score = pred[idx]
                output_row = [name] + score
                output_data.append(output_row)

                if visualize:
                    print(name)
                    current_att = np.round(att[idx][0].cpu().numpy(), 2)

                    fig, ax = plt.subplots(nrows=1, ncols=2)
                    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)
                    # cb = ax[1].imshow(current_att, cmap='gray')
                    img = Image.open(os.path.join(test_dir, name)).resize((302, 227))
                    # img =
                    ax[0].imshow(img)
                    ax[1].imshow(img)
                    for i in range(1, 10):
                        ax[1].plot([(302/10)*i, (302/10)*i], [0, 225], color='black', linewidth=1, alpha=0.8)
                    for i in range(1, 8):
                        ax[1].plot([0, 300], [(227/8)*i, (227/8)*i], color='black', linewidth=1, alpha=0.8)

                    current_att = np.round(current_att * 100)
                    att_max = np.max(current_att)
                    # att_min = np.min(current_att[current_att != 0])
                    # color_map = cm.get_cmap('cool', lut=att_max)
                    color_map = cm.get_cmap('PuRd', lut=att_max)

                    for j in range(8):
                        for i in range(10):
                            att_i_j = int(current_att[j][i])
                            if att_i_j > 0:
                                ax[1].text(i*(302/10)+10, j*(227/8)+18, s=f'{att_i_j}',
                                           color='yellow',
                                           bbox=dict(facecolor=color_map(att_i_j/att_max),
                                                     alpha=0.8))
                    # tb = ax[0].table(cellText=current_att, loc=(0, 0), cellLoc='center')

                    # tc = tb.properties()['child_artists']
                    # for cell in tc:
                    #     cell.set_height(1 / 8)
                    #     cell.set_width(1 / 10)

                    pred_class = LABELS[np.argmax(score)]
                    # fig.suptitle(f'{name}\n{pred_class}\n{np.round(score, 3)}')
                    # fig.colorbar(cb, ax=ax[1])
                    fig.canvas.set_window_title(f'{pred_class}_{name}')
                    ax[0].axis('off')
                    ax[1].axis('off')
                    plt.show()
                    plt.close()

    output_df = pd.DataFrame(data=output_data,
                             columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    output_df.sort_values(by=['img'], inplace=True)
    output_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    import os
    from attention_model import ResNetAttention1, ResNetAttention2, ResNetAttention3, ResNetAttention4
    root_model_folder = 'D:\gmu\sem-4\CS747\outputs'
    # ['r18-relu-533894', 'r18-relu-fc-533906', 'r18-soft-534033', 'r18-soft-ent-534037',
    # 'r18-soft-fc-533906']

    model_folder = 'r18-soft-ent-534037'
    model = ResNetAttention2()

    # model_folder = 'r18-relu-533894'
    # model = ResNetAttention1()

    # model_folder = 'r18-soft-534033'
    # model = ResNetAttention2()

    model_ckpt_folder = os.path.join(root_model_folder, model_folder)
    for file in os.listdir(model_ckpt_folder):
        # if 'val' in file:                           # Best val
        if '.pt' in file and 'val' not in file:     # Final Train
            model_ckpt = os.path.join(model_ckpt_folder, file)
            break
    print(model_ckpt)
    submission_path = os.path.join('submissions', file[:-3] + '.csv')

    state_dict = torch.load(os.path.join('models', model_ckpt))
    test_model(model=model, state_dict=state_dict, save_path=submission_path, visualize=True)
