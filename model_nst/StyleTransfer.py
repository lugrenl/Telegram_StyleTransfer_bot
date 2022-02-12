import copy
import torch
import logging
import torch.nn as nn
from io import BytesIO
import torch.optim as optim
from PIL import Image, ImageOps
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


logging.basicConfig(level=logging.INFO)


class ImageProcessing:
    def __init__(self, new_size, device):
        self.new_size = new_size
        self.device = device
        self.image_size = None

    def image_loader(self, image_name):
        image = Image.open(image_name)
        self.image_size = image.size
        image = ImageOps.pad(image, (self.new_size, self.new_size))
        loader = transforms.ToTensor()
        image = loader(image).unsqueeze(0)

        return image.to(self.device, torch.float)

    def get_image(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        unloader = transforms.ToPILImage()
        image = unloader(image)
        image = ImageOps.fit(image, self.image_size)

        # transform PIL image to send to telegram
        bio = BytesIO()
        bio.name = 'output.jpeg'
        image.save(bio, 'JPEG')
        bio.seek(0)

        return bio


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value, not a variable.
        # Otherwise, the forward method of the criterion will throw an error.
        self.target = target.detach()  # константа, убираем ее из дерева вычислений
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    @staticmethod
    def gram_matrix(input):
        batch_size, h, w, f_map_num = input.size()  # batch size(=1)
        # (h,w) = dimensions of a feature map (N=h*w)
        features = input.view(batch_size * h, w * f_map_num)  # resize F_XL into \hat F_XL
        g = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return g.div(batch_size * h * w * f_map_num)

    def forward(self, input):
        g = self.gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleTransfer:
    def __init__(self, num_steps, device='cpu', style_weight=100000, content_weight=1):
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.device = device

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(self, style_img, content_img):

        cnn = models.vgg19(pretrained=False, progress=False)
        cnn = copy.deepcopy(cnn.features[:11])
        
        cnn.load_state_dict(torch.load("models_wts/vgg19.pth", map_location=torch.device(self.device)))
        cnn = cnn.to(self.device).eval()

        normalization = Normalization(self.device).to(self.device)

        content_losses = []  # just in order to have an iterable access to or list of content/style
        style_losses = []  # losses

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)  # Переопределим relu уровень
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        # добaвляет содержимое тензора картинки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def transfer_style(self, style_img, content_img):
        input_img = content_img.clone()
        model, style_losses, content_losses = self.get_style_model_and_losses(
            style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= self.num_steps:

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ощибки
                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                if run[0] % 50 == 0:
                    logging.info(f"run: {run[0]}")
                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img


def run_nst(style_image, content_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_processing = ImageProcessing(new_size=256, device=device)
    content_processing = ImageProcessing(new_size=256, device=device)

    style_image = style_processing.image_loader(style_image)
    content_image = content_processing.image_loader(content_image)

    transfer = StyleTransfer(num_steps=200, device=device)
    output = transfer.transfer_style(style_image, content_image)
    output = content_processing.get_image(output)

    return output
