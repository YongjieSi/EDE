import copy
import logging
import numpy as np
import torch
from torch import nn
import torchaudio
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
from models.ast_models import ASTModel
from models.vit.vision_transformer_memo import Specialized_Vit
# from backbone.ast_backbone_adapter import Specialized_Vit

from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from easydict import EasyDict



def add_backbone(args, pretrained=False):
    name = args["add_backbone_type"].lower()

    model = Specialized_Vit()
    model.out_dim = 768
    return model

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()

    if name == "ast_audioset_10_10_0.4593" :
        model = ASTModel(input_tdim=201, label_dim=527, imagenet_pretrain  = True,audioset_pretrain=True)
        return model.eval()



class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]
        self.args = args
        self.fc_for_train = None

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'
        self.set_fea_extractor()
            

    @property
    def feature_dim(self):
        return self.backbone.out_dim
        # return 512
        
    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x
        
    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        
    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self



class MyAdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(MyAdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor = get_backbone(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.intit_extractor = get_backbone(args, pretrained) 
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args
        self.cos_fc=None
        self.set_fea_extractor()

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
    
    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x
    
    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2) #[batch_size, 1, time_steps, mel_bins]
        base_feature_map = self.TaskAgnosticExtractor(x) #[batch_size, 1, time_steps, mel_bins]
        
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        # out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
    def forward_base(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2) #[batch_size, 1, time_steps, mel_bins]
        base_feature_map = self.TaskAgnosticExtractor(x) #[batch_size, 1, time_steps, mel_bins]
        out=self.fc(base_feature_map) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(base_feature_map[:,-768:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":base_feature_map})
        return out
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
    def _forward(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2) #[batch_size, 1, time_steps, mel_bins]
        init_feature  = self.intit_extractor(x)
        init_feature = self.TaskAgnosticExtractor.v.norm(init_feature)
        init_feature = init_feature.mean(1)
        base_feature_map = self.TaskAgnosticExtractor(x) #[batch_size, 1, time_steps, mel_bins]
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        features = torch.cat((init_feature,features), 1)
        out=self.cos_fc(features) #{logits: self.fc(features)}
        out.update({"features":features})
        return out
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
    def update_fc(self,nb_classes):
        _new_extractor = add_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        cos_fc = self.generate_cosfc(self.feature_dim + 768, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        if self.cos_fc is not None:
            nb_output = self.cos_fc.out_features
            weight = copy.deepcopy(self.cos_fc.weight.data)
            cos_fc.weight.data[:nb_output,:self.feature_dim+768-self.out_dim] = weight
        del self.fc
        del self.cos_fc
        
        self.fc = fc
        self.cos_fc = cos_fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def update_fc_base(self,nb_classes):
        fc = self.generate_fc(768, nb_classes)
        cos_fc = self.generate_cosfc(768 + 768, nb_classes)
        del self.fc
        self.fc = fc
        self.cos_fc = cos_fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(768,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        # fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def generate_cosfc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    
