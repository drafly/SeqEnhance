import torch
import isp.EE as EE

def isp_execute(image, ee_amount):
    '''
    INPUT: (BHWC)
    OUTPUT: (BHWC)
    '''
    
    out = EE.begin_EE(image, ee_amount)
    
    return out

if __name__=="__main__":
    input = torch.rand(1, 3, 256, 256).cuda()
    output = isp_execute(input, 1)
    print(output.shape)