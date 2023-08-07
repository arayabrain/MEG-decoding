from .sc_mbm import SCMBMTrainer, MAEforEEG

def get_model_and_trainer(config, device_count=1, usewandb=True, only_model:bool=False):
    if 'model' in config:
        # model_name = config.model.name
        model_config = config.model
    else:
        # model_name = config.meg_encoder.name
        model_config = config.meg_encoder
    training_config = config.training
    model_name = model_config.name
    if model_name == 'sc_mbm':
        model = MAEforEEG(**model_config.parameters)
        trainer = SCMBMTrainer(training_config, device_count, usewandb) if not only_model else None
    else:
        raise ValueError('model_config.name {} is not supported'.format(model_config.name))

    return model, trainer