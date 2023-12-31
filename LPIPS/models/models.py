'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''

def create_model(opt):
    model = None
    print(opt.model)
    from .siam_model import *
    model = DistModel()
    model.initialize(opt, opt.batchSize, )
    print("model [%s] was created" % (model.name()))
    return model

