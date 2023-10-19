import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from skimage.draw import polygon2mask
import PIL
from pathlib import Path
import os
import sys
import json
import gc
# Custom modules
import dnnlib
import legacy
from training.networks import AugmentedGenerator


class Interpreter(pl.LightningModule):
    """
    Three-layer feed-forward neural network
    Used to learn mapping from hidden representations to labels
    Details in https://arxiv.org/pdf/2104.06490.pdf
    """
    def __init__(self, input_dim=256, hidden=[128, 32], num_outputs=1,
                 loss_mode="mse", lr=1e-3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden[0]),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden[1]),
            nn.Linear(hidden[1], num_outputs))
        self.loss_mode = loss_mode
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.mlp(x).squeeze()
        if self.loss_mode=="mse":
            loss = nn.functional.mse_loss(y_hat, y)
        elif self.loss_mode=="crossentropy":
            loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.mlp(x).squeeze()


def get_raw_labels(seed:int, annotations:dict):
    """
    Convert manual annotations of a given synthetic image into torch.Tensor
    Annotation dictionary is in COCO format
    Detects and returns the annotation type
    """
    def clip_coordinate(c, vmin, vmax):
        c = max(c, vmin)
        c = min(c, vmax - 1)
        return c

    # Sanity Check
    error_msg = f"Provided seed={seed} is not found in annotation file!"
    img_dict = {img["file_name"]: img for img in annotations["images"]}
    assert f'seed{seed:04d}.png' in img_dict.keys(), error_msg

    img = img_dict[f'seed{seed:04d}.png']
    image_id = img["id"]
    width, height = img["width"], img["height"]
    
    # Find out how the images where annotated
    for a in annotations["annotations"]:
        if "keypoints" in a:
            mode = "landmarks"
            break
        if "segmentation" in a:
            mode = "masks"
            break
    max_label = max(c["id"]  for c in annotations["categories"])

    labels = torch.zeros((width, height), dtype=torch.int32)
    for a in annotations["annotations"]:
        c = a["category_id"]
        if a["image_id"] == image_id:
            if "keypoints" in a:
                j, i = a["keypoints"][:2]
                i = clip_coordinate(i, 0, height)
                j = clip_coordinate(j, 0, width)
                labels[i, j] = c
                continue
            if "segmentation" in a:
                polygon = np.array(a["segmentation"]).reshape(-1, 2)
                polygon = polygon[:, [1, 0]]
                labels += c * polygon2mask((height, width), polygon)
    
    # Prevent overflow
    labels[labels > max_label] = 0

    return labels, mode


def get_gaussian_heatmap(labels: torch.Tensor):
    """
    Convert landmarks tensor into a Gaussian Heat Map
    Input should be a 2D/3D torch.Tensor containing only zeros and ones
    """
    sigma = max(np.log2(labels.size(-1) / 16), 1)
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    ghm = torch.zeros_like(labels)
    for idx, labels2d in enumerate(labels):
        for x0, y0 in torch.nonzero(labels2d):
            x, y = torch.meshgrid(
                torch.linspace(0, labels2d.size(0)-1, labels2d.size(0)) - x0,
                torch.linspace(0, labels2d.size(1)-1, labels2d.size(1)) - y0,
                indexing='ij'
            )
            heatmap = torch.exp(-((x**2 + y**2) / (2 * sigma**2)))
            heatmap = heatmap / heatmap.sum()
            ghm[idx] += heatmap
    return ghm.squeeze()


def preprocess_labels(labels: torch.Tensor, mode="masks", multiclass=False):
    """
    Prepare labels for Interpreter training
    """
    if multiclass and mode == "masks":
        # 2D array containing integers 0 for background, 1 for class_1, etc.
        labels = labels.to(torch.long)
    else:
        if multiclass:
            # 3D tensor containing zeros and ones
            labels3d = torch.arange(1, 1 + labels.int().max().item())
            labels3d = labels3d[:, None, None] == labels
            labels = labels3d.to(torch.float)
        else:
            # 2D array containing zeros and ones
            labels = (labels > 0).to(torch.float)
        if mode == "landmarks":
            # Transform Landmarks Into Gaussian Heat Map
            labels = get_gaussian_heatmap(labels)
        # Rescale to -1..1
        if labels.max() > 0:
            labels = 2 * labels / labels.max() - 1
        else:
            labels[labels==0] = -1.0
    return labels


@torch.no_grad()
def get_features(augmented_generator, z, label, truncation_psi, noise_mode,
                 memory_constrained=True):
    """
    Use modified StyleGAN generator to produce an image 
    and the corresponding intermediate features (per pixel)
    """
    img, acts = augmented_generator(z, label, truncation_psi=truncation_psi,
                                    noise_mode=noise_mode,
                                    memory_constrained=memory_constrained)

    # Upsample to largest resolution
    size = acts[-1].shape[2:] 
    for act_idx, act in enumerate(acts):
        act = torch.nn.functional.interpolate(act, size, mode='bilinear',
                                              align_corners=None)
        acts[act_idx] = act
    
    # Concatenate along channel dimension
    features = torch.cat(acts, dim=1)

    return img, features


def flatten_features(features: torch.Tensor):
    bs, vector_dim, h, w = features.shape
    features = features.permute(0, 2, 3, 1) # NCHW > NHWC
    features = features.reshape(bs*h*w, vector_dim)
    return features, bs, h, w


def convert_tensor_to_pil(img: torch.Tensor, torgb=True):
    """
    Convert synthetic image to PIL.Image
    """
    assert img.ndim == 3, f"Expected 3 dimensions, got: {img.ndim}"
    num_channels = img.size(0)
    img = img.permute(1, 2, 0) # CHW > HWC
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.squeeze().cpu().numpy()
    if num_channels == 3:
        img_pil = PIL.Image.fromarray(img, 'RGB')
    elif num_channels == 1:
        img_pil = PIL.Image.fromarray(img, 'L')
        if torgb: 
            img_pil = img_pil.convert("RGB")
    else:
        print(f"Images with {num_channels} channels are not supported!")
        sys.exit(1)
    return img_pil


def convert_mask_to_pil(lbl: torch.Tensor, palette=None):
    """
    Convert label tensor to PIL.Image
    """
    assert lbl.ndim in [2, 3], f"Expected 2 or 3 dimensions, got: {lbl.ndim}"
    make_npuint8 = lambda a: a.cpu().numpy().astype(np.uint8)
    
    # Convert one-hot 3D tensor into 2D integer array
    if lbl.ndim == 3:
        lbl_2d = torch.zeros(lbl.shape[1:])
        for i, l in enumerate(lbl, 1):
            lbl_2d[l > 0.5] = i
        lbl = lbl_2d
    
    # Use color palette
    if palette is not None:
        msk = make_npuint8(lbl)
        msk_pil = PIL.Image.fromarray(msk, "L")
        msk_pil.putpalette(palette)
        msk_pil = msk_pil.convert("RGB")
    # Change from grayscale to green
    else:
        msk = make_npuint8(lbl * 255)
        blank = np.zeros_like(msk)
        msk = np.stack([blank, msk, blank], axis=-1)
        msk_pil = PIL.Image.fromarray(msk, "RGB")
    
    return msk_pil


def make_composite(img: torch.Tensor, lbl: torch.Tensor, palette=None):
    """
    Overlay labels on top of synthetic image
    """
    img_pil = convert_tensor_to_pil(img, torgb=True)
    msk_pil = convert_mask_to_pil(lbl, palette)
    alpha = PIL.Image.new("L", img_pil.size, 128)
    composite = PIL.Image.composite(img_pil, msk_pil, alpha)
    return img_pil, composite


def prepare_training_data(training_seeds, annotations, augmented_generator,
                          device, label, verbose, truncation_psi, noise_mode,
                          memory_constrained, result_dir, multiclass=False):
    """
    Prepare training data for the interpreter
    """
    result_dir = Path(result_dir)
    (result_dir / 'groundtruth').mkdir(parents=True, exist_ok=True)
    
    mode = get_raw_labels(training_seeds[0], annotations)[1]
    print(f'Using {mode} from annotation file!') 

    if multiclass:
        num_classes = len(annotations["categories"])
        palette = np.concatenate([
            [0, 0, 0],
            np.random.randint(0, 256, 3*num_classes, dtype=np.uint8)
            ]).tolist()
    else:
        palette = None

    z_dim = augmented_generator.z_dim
    c_dim = augmented_generator.c_dim

    X, y = [], []
    for seed_idx, seed in enumerate(training_seeds, 1):
        if verbose:
            print('Generating training data for seed ', end='')
            print('%d (%d/%d) ...' % (seed, seed_idx, len(training_seeds)))

        z = np.random.RandomState(seed).randn(1, z_dim)
        z = torch.from_numpy(z).to(device)

        lbl = torch.zeros([1, c_dim], device=device)
        if c_dim:
            if label.sum() == 0:
                lbl[0, seed % c_dim] = 1
            else:
                lbl = label
            
        img, features = get_features(augmented_generator, z, lbl,
                                     truncation_psi=truncation_psi,
                                     noise_mode=noise_mode,
                                     memory_constrained=memory_constrained)

        lbl, mode = get_raw_labels(seed, annotations)
        lbl = preprocess_labels(lbl, mode, multiclass)
     
        # Flatten features and labels
        features, _, h, w = flatten_features(features)
        if (multiclass and mode == 'landmarks'):
            labels = lbl.permute(1, 2, 0) # CHW > HWC
            labels = labels.reshape(h*w, -1) 
        else :
            labels = lbl.reshape(-1)

        X.append(features)
        y.append(labels)

        # Save composite for visual inspection
        lbl = lbl if (multiclass and mode == 'masks') else (lbl + 1) / 2 
        img_pil, composite = make_composite(img[0], lbl, palette)

        sv_dir = result_dir / 'groundtruth' 
        composite.save(sv_dir / f'seed{seed:04d}-annot.png')
        img_pil.save(sv_dir / f'seed{seed:04d}.png')

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    if verbose:
        label_dim = y.size(1) if y.dim() == 2 else 1
        print(f'num_training_samples = {X.size(0)}')
        print(f'vector_dim = {X.size(1)}')
        print(f'label_dim = {label_dim}')
 
    return X, y, palette


def train_interpreter(X: torch.Tensor, y: torch.Tensor, multiclass=False,
                      batch_size=64, max_epochs=2):
    """
    Train interpreter using the given configuration
    """
    vector_dim = X.size(1)
    label_dim = y.size(1) if y.dim() == 2 else 1
    
    dataset = torch.utils.data.TensorDataset(X.cpu(), y.cpu())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=os.cpu_count(),
                                               shuffle=True)
    trainer = pl.Trainer(max_epochs=max_epochs)

    if multiclass and label_dim == 1:
        num_classes = len(y.unique())
        model = Interpreter(input_dim=vector_dim, num_outputs=num_classes,
                            loss_mode="crossentropy")
    else:
        model = Interpreter(input_dim=vector_dim, num_outputs=label_dim,
                            loss_mode="mse")
    
    trainer.fit(model=model, train_dataloaders=train_loader)
    
    return model


@torch.no_grad()
def generate_labeled_images(seeds: list, augmented_generator, interpreter,
                            device, label, verbose, truncation_psi, noise_mode,
                            memory_constrained, save_dir, result_dir, palette, 
                            num_samples, training_seeds):
    """
    Generate a batch of synthetic images and corresponding labels
    """
    result_dir = Path(result_dir)
    (result_dir / 'groundtruth').mkdir(parents=True, exist_ok=True)
    (result_dir / 'unseen').mkdir(parents=True, exist_ok=True)

    multiclass = palette is not None
    z_dim = augmented_generator.z_dim
    c_dim = augmented_generator.c_dim
    
    z = np.concatenate(
        [np.random.RandomState(seed).randn(1, z_dim) for seed in seeds]
    )
    z = torch.from_numpy(z).to(device)
    
    labels = torch.zeros([len(seeds), c_dim], device=device)
    if c_dim:
        for idx, seed in enumerate(seeds):
            if label.sum() == 0:
                labels[idx, seed % c_dim] = 1
            else:
                labels[idx] = label 

    # Generate synthetic images and corresponding hidden representations
    imgs, features = get_features(augmented_generator, z, labels,
                                  truncation_psi=truncation_psi,
                                  noise_mode=noise_mode,
                                  memory_constrained=memory_constrained)

    # Use Interpreter to create labels
    features = features.to(device)
    interpreter = interpreter.eval().to(device)
    features, bs, h, w = flatten_features(features)
    predictions = interpreter(features)

    # Put labels in correct format
    if multiclass:
        predictions = predictions.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
    else:
        predictions = predictions.reshape(bs, h, w)
    
    if multiclass and interpreter.loss_mode == 'crossentropy':
        predictions = predictions.softmax(dim=1).argmax(dim=1)
    else:
        predictions = predictions.clamp(-1, 1) 
        predictions = ((predictions + 1) / 2).clamp(0, 1)

    for idx, seed in enumerate(seeds):
        img = imgs[idx]
        preds = predictions[idx]
        img_pil = convert_tensor_to_pil(img, torgb=False)
        
        # Save synthetic image and corresponding predictions
        img_pil.save(f'{save_dir}/seed{seed:04d}.png')
        np.save(f'{save_dir}/seed{seed:04d}.npy', preds.cpu().numpy())

        # Save composite for visual inspection
        _, composite = make_composite(img, preds, palette)
        if seed in training_seeds:
            sv_path = result_dir / 'groundtruth' / f'seed{seed:04d}-pred.png'
            composite.save(sv_path)
        elif seed % 100 == 0:
            if verbose: print('(%d/%d) ...' % (seed, num_samples))
            sv_path = result_dir / 'unseen' / f'seed{seed:04d}-pred.png'
            composite.save(sv_path)


def initial_setup(verbose, seeds, class_idx, network_pkl, annotation_file, 
                  multiclass, result_dir, save_dir):
    # Load COCO annotations
    if Path(annotation_file).exists():
        with open(annotation_file) as f:
            annotations = json.load(f)
    else:
        print(f'Annotation file: {annotation_file} not found! Leaving ...')
        sys.exit(1)
    
    # Load generator network and convert to an augmented copy
    if verbose: print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    c_dim = G.c_dim
    augmented_generator = AugmentedGenerator(G).to(device)
    del G

    # Detect all seeds from annotation file
    if seeds is None:
        annotated_ids = {a["image_id"] for a in annotations["annotations"]}
        assert len(annotated_ids), "No annotation found in file !"
        seeds = [int(img["file_name"][4:8]) 
                 for img in annotations["images"]
                 if img["id"] in annotated_ids]
        print("Using all images detected in annotation file!")
        print(f"Seeds: {str(seeds).lstrip('[').rstrip(']').replace(' ','')}")

    # Check class labels
    if  len(annotations["categories"]) == 1:
        print("warn: --multiclass ignored")
        multiclass = False  

    # Prepare desired image-level labels / placeholder
    label = torch.zeros([1, c_dim], device=device)
    if c_dim != 0:
        if class_idx is None:
            print('warn: generate all class labels when ' \
                  'using a conditional network and --class not specified')
        else:
            label[0, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when ' \
                  'using an unconditional network')

    gc.collect()
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    return seeds, annotations, augmented_generator, device, label, multiclass
