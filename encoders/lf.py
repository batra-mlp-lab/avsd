import torch
from torch import nn
from torch.nn import functional as F

from utils import DynamicRNN


class LateFusionEncoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-img_feature_size', default=4096, help='Channel size of image feature')
        parser.add_argument('-vid_feature_size', default=4096, help='Channel size of video feature')
        parser.add_argument('-audio_feature_size', default=4096, help='Channel size of audio feature')
        parser.add_argument('-embed_size', default=300, help='Size of the input word embedding')
        parser.add_argument('-rnn_hidden_size', default=512, help='Size of the multimodal embedding')
        parser.add_argument('-num_layers', default=2, help='Number of layers in LSTM')
        parser.add_argument('-max_history_len', default=60, help='Size of the multimodal embedding')
        parser.add_argument('-dropout', default=0.5, help='Dropout')
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.word_embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0)
        
        if 'dialog' in args.input_type or 'caption' in args.input_type:
            self.hist_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                    batch_first=True, dropout=args.dropout)
            self.hist_rnn = DynamicRNN(self.hist_rnn)
        
        self.ques_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                batch_first=True, dropout=args.dropout)
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        self.dropout = nn.Dropout(p=args.dropout)
        # fusion layer
        if args.input_type == 'question_only':
            fusion_size = args.rnn_hidden_size
        if args.input_type == 'question_dialog':
            fusion_size =args.rnn_hidden_size * 2
        if args.input_type == 'question_audio':
            fusion_size =args.rnn_hidden_size + args.audio_feature_size
        if args.input_type == 'question_image' or args.input_type=='question_video':
            fusion_size = args.img_feature_size + args.rnn_hidden_size 
        if args.input_type == 'question_caption_image' or args.input_type=='question_dialog_video' or args.input_type=='question_dialog_image':
            fusion_size = args.img_feature_size + args.rnn_hidden_size * 2
        if args.input_type == 'question_video_audio':
            fusion_size = args.img_feature_size + args.rnn_hidden_size + args.audio_feature_size
        if args.input_type == 'question_dialog_video_audio':
            fusion_size = args.img_feature_size + args.rnn_hidden_size * 2 + args.audio_feature_size
        
        self.fusion = nn.Linear(fusion_size, args.rnn_hidden_size)

        if args.weight_init == 'xavier':
            nn.init.xavier_uniform(self.fusion.weight.data)
        elif args.weight_init == 'kaiming':
            nn.init.kaiming_uniform(self.fusion.weight.data)
        nn.init.constant(self.fusion.bias.data, 0)

    def forward(self, batch):
        if 'image' in self.args.input_type:
            img = batch['img_feat']
            # repeat image feature vectors to be provided for every round
            img = img.view(-1, 1, self.args.img_feature_size)
            img = img.repeat(1, self.args.max_ques_count, 1)
            img = img.view(-1, self.args.img_feature_size)
        
        if 'audio' in self.args.input_type:
            audio = batch['audio_feat']
            # repeat audio feature vectors to be provided for every round
            audio = audio.view(-1, 1, self.args.audio_feature_size)
            audio = audio.repeat(1, self.args.max_ques_count, 1)
            audio = audio.view(-1, self.args.audio_feature_size)

        if 'video' in self.args.input_type:
            vid = batch['vid_feat']
            # repeat image feature vectors to be provided for every round
            vid = vid.view(-1, 1, self.args.vid_feature_size)
            vid = vid.repeat(1, self.args.max_ques_count, 1)
            vid = vid.view(-1, self.args.vid_feature_size)
        
        if 'dialog' in self.args.input_type or 'caption' in self.args.input_type:
            hist = batch['hist']
            # embed history
            hist = hist.view(-1, hist.size(2))
            hist_embed = self.word_embed(hist)
            hist_embed = self.hist_rnn(hist_embed, batch['hist_len'])
        
        ques = batch['ques']

        # embed questions
        ques = ques.view(-1, ques.size(2))
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_embed, batch['ques_len'])
        
        if self.args.input_type == 'question_only':
            fused_vector = ques_embed
        if self.args.input_type == 'question_dialog':
            fused_vector = torch.cat((ques_embed, hist_embed), 1)
        if self.args.input_type == 'question_audio':
            fused_vector = torch.cat((audio, ques_embed), 1)
        if self.args.input_type == 'question_image':
            fused_vector = torch.cat((img, ques_embed), 1)
        if self.args.input_type=='question_video':
            fused_vector = torch.cat((vid, ques_embed), 1)
        if self.args.input_type=='question_dialog_image':
            fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        if self.args.input_type == 'question_dialog_video':
            fused_vector = torch.cat((vid, ques_embed, hist_embed), 1)
        if self.args.input_type == 'question_caption_image':
            fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        if self.args.input_type == 'question_video_audio':
            fused_vector = torch.cat((vid, audio, ques_embed), 1)
        if self.args.input_type == 'question_dialog_video_audio':
            fused_vector = torch.cat((vid, audio, ques_embed, hist_embed), 1)
 
        fused_vector = self.dropout(fused_vector)

        fused_embedding = F.tanh(self.fusion(fused_vector))
        return fused_embedding
