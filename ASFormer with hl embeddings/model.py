import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math
from tqdm import tqdm

from eval import segment_bars_with_confidence

LOW_RESOLUTION_DIM = 2048
HIGH_RESOLUTION_DIM = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        nb = math.ceil(L / self.bl)
        pad_len = self.bl * nb - L
        L_pad = L + pad_len

        # Pad q, k, v, and mask
        q = F.pad(q, (0, pad_len))
        k = F.pad(k, (0, pad_len))
        v = F.pad(v, (0, pad_len))
        padding_mask = F.pad(mask[:, 0:1, :], (0, pad_len))

        # Reshape for attention
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(-1, c1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(-1, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(-1, c3, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(-1, 1, self.bl)

        # Attention
        output, _ = self.att_helper.scalar_dot_att(q, k, v, padding_mask)  # output: [B*nb, v_dim // r3, bl]
        output = self.conv_out(F.relu(output))  # output: [B*nb, v_dim, bl]

        # Now get updated c3 value from output
        _, c3_out, _ = output.size()

        # Correct reshape using real c3_out
        try:
            output = output.reshape(m_batchsize, nb, c3_out, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3_out, L_pad)
        except Exception as e:
            raise RuntimeError(
                f"Invalid reshape in _block_wise_self_att: m_batchsize={m_batchsize}, nb={nb}, "
                f"c3_out={c3_out}, bl={self.bl}, expected={m_batchsize * nb * c3_out * self.bl}, "
                f"actual={output.numel()}"
            ) from e

        output = output[:, :, :L]  # crop padding
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

class OriginalEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_f_maps):
        super(OriginalEncoder, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AttModule(2 ** i, num_f_maps, num_f_maps, r1=2, r2=2, att_type='block_att', stage='encoder', alpha=1.0)
            for i in range(num_layers)
        ])

    def forward(self, x, mask):
        out = self.input_proj(x)
        for layer in self.layers:
            out = layer(out, None, mask)
        return out

class Predictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Predictor, self).__init__()
        self.fusion_layer = nn.Conv1d(input_dim, num_classes, 1)

    def forward(self, x, mask):
        return self.fusion_layer(x) * mask[:, 0:1, :]
    
class HighResCNNEncoder(nn.Module):
    def __init__(self, input_dim, num_f_maps):
        super(HighResCNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, num_f_maps, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1)
        )

    def forward(self, x, mask):
        return self.cnn(x) * mask[:, 0:1, :]

class DualEncoderASFormer(nn.Module):
    def __init__(self, num_layers, num_f_maps, num_classes):
        super(DualEncoderASFormer, self).__init__()
        self.encoder_low = OriginalEncoder(num_layers=num_layers, input_dim=LOW_RESOLUTION_DIM, num_f_maps=num_f_maps)
        self.encoder_high = HighResCNNEncoder(input_dim=HIGH_RESOLUTION_DIM, num_f_maps=num_f_maps)
        self.fusion_predictor = Predictor(input_dim=num_f_maps, num_classes=num_classes)
        self.decoder_input_proj = nn.Conv1d(num_classes, num_f_maps, 1)  # Fix here: convert class scores to num_f_maps before decoder
        self.decoders = nn.ModuleList([
            Decoder(num_layers, 2, 2, num_f_maps, num_f_maps, num_classes, att_type='block_att', alpha=exponential_descrease(s))
            for s in range(3)
        ])

    def pad_features(self, x, target_dim):
        # x: [batch, channels, seq_len]
        if x.size(1) < target_dim:
            pad_size = target_dim - x.size(1)
            pad_tensor = torch.zeros(x.size(0), pad_size, x.size(2)).to(x.device)
            x = torch.cat([x, pad_tensor], dim=1)
        return x

    def forward(self, x_low, x_high, mask):
        x_high = self.pad_features(x_high, HIGH_RESOLUTION_DIM)

        # print("x_low shape: ", x_low.shape)
        # print("x_high shape: ", x_high.shape)
        # print("mask shape: ", mask.shape)

        feat_low = self.encoder_low(x_low, mask)
        # print("feat_low shape after encoder_low:", feat_low.shape)

        feat_high = self.encoder_high(x_high, mask)
        # print("feat_high shape after encoder_high:", feat_high.shape)

        scores = self.fusion_predictor(feat_low, mask)
        # print("scores shape after fusion predictor:", scores.shape)

        outputs = [scores.unsqueeze(0)]
        projected_scores = self.decoder_input_proj(scores)
        # print("projected_scores shape for decoder input:", projected_scores.shape)

        feature =feat_high

        for decoder in self.decoders:
            scores, feature = decoder(F.softmax(projected_scores, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs.append(scores.unsqueeze(0))

        return torch.cat(outputs, dim=0)


 
class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        print('Initializing Model...')
        self.model = DualEncoderASFormer(num_layers, num_f_maps, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}'.format(learning_rate))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                x_low, x_high, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                x_low, x_high = x_low.to(device), x_high.to(device)
                batch_target, mask = batch_target.to(device), mask.to(device)

                optimizer.zero_grad()
                # print("x_low shape: ", x_low.shape)
                # print("x_high shape: ", x_high.shape)
                ps = self.model(x_low, x_high, mask)

                loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            scheduler.step(epoch_loss)
            batch_gen.reset()
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total))

            if (epoch + 1) % 10 == 0 and batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False
        with torch.no_grad():
            while batch_gen_tst.has_next():
                x_low, x_high, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                x_low, x_high = x_low.to(device), x_high.to(device)
                batch_target, mask = batch_target.to(device), mask.to(device)
                p = self.model(x_low, x_high, mask)
                _, predicted = torch.max(p.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

        acc = float(correct) / total
        print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))
        self.model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            batch_gen_tst.reset()
            import time
            time_start = time.time()
            while batch_gen_tst.has_next():
                x_low, x_high, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                x_low, x_high = x_low.to(device), x_high.to(device)
                batch_target, mask = batch_target.to(device), mask.to(device)
                predictions = self.model(x_low, x_high, mask)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    batch_target = batch_target.squeeze()
                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                with open(results_dir + "/" + f_name, "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(' '.join(recognition))
