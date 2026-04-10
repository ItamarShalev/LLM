ATTENTION_HEADS = {}

def attention_hook(name):
    def hook(model, input, output):
        # input is a tuple, and we want the first element of 
        ATTENTION_HEADS[name] = output.detach().cpu()
    return hook

