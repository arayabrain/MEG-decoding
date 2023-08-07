from transformers import CLIPProcessor, CLIPModel

def get_image_encoder(model_name:str, parameters:dict):
    if model_name == 'vit_clip':
        model = CLIPModel.from_pretrained(parameters['path'])
        preprocess = CLIPProcessor.from_pretrained(parameters['path'])
        return model, preprocess
    else:
        raise NotImplementedError(f'{model_name} is not implemented')