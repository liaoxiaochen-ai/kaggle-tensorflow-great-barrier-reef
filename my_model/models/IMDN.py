import torch.nn as nn
import basicblock as B
import time
import numpy as np
import torch


"""
# --------------------------------------------
# simplified information multi-distillation
# network (IMDN) for SR
# --------------------------------------------
References:
@inproceedings{hui2019lightweight,
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}
@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}
# --------------------------------------------
"""


# --------------------------------------------
# modified version, https://github.com/Zheng222/IMDN
# first place solution for AIM 2019 challenge
# --------------------------------------------

class IMDN_base(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_base, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='C'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_part(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_part, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_wide(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_wide, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='c', negative_slope=negative_slope) for _ in range(nb-4)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_dsc(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_dsc, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='c', negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_RFDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_RFDN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_RFDN(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class RFDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(RFDN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_RFDN(nc, nc, mode='C'+act_mode, negative_slope=negative_slope) for _ in range(4)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_Mix(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_Mix, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_Mix(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_dropout(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05, states='test'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_dropout, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        self.states = states
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)))
        self.upsample = B.sequential(*m_uper)

    def dropout(self, x, probability=0.1):
            n_channels = x.shape[1]
            p = np.random.binomial(1, 1-probability, n_channels)
            p = torch.tensor(np.expand_dims(p, (1,2))).to(x.device)
            x = torch.multiply(x, p)
            return x

    def forward(self, x):
        x = self.model(x)
        if self.states == 'train':
            x = self.dropout(x, probability=0.02)
        x = self.upsample(x)
        return x

class IMDN_DDF(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_DDF, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_DDF(nc, nc, mode='D'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_DDFUP(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='ddfup', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN_DDFUP, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
            m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        elif upsample_mode == 'ddfup':
            upsample_block = B.upsample_ddfup
            m_uper = upsample_block(in_channels=64)
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_all(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        super(IMDN_all, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='c')
        m_body = [B.IMDBlock(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='c'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_OFA(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        super(IMDN_OFA, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_weight_OFA(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_weight(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        super(IMDN_weight, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_weight_attention(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_activation(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        super(IMDN_activation, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_activation_attention(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

class IMDN_SE(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        super(IMDN_SE, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock_SE_attention(nc, nc, mode='c'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    mdel
