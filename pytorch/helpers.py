def deactivate_layer (model, layer_name):
    for p in getattr(model, layer_name).parameters():
        p.requires_grad = False