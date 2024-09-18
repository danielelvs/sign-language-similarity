import argparse
import yaml

class Parser:
  """
  Parser class for handling command-line arguments and configuration files for the Siamese Network One-Shot Learning.
  """

  def __init__(self):
    """
    Initializes the argument parser for the Siamese Network One-Shot Learning with dynamic parameters.
    The parsed arguments are stored in the `self.args` attribute.
    """
    parser = argparse.ArgumentParser(description='Siamese Network One-Shot Learning with dynamic parameters')

    # config file path
    parser.add_argument('-c', '--config', type=str, help='Config file path', default='config/default.yaml',)

    # model parameters
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('-is', '--input_size', type=int, help="Input size of the model", default=28)
    model_group.add_argument('-hs', '--hidden_size', type=int, help="Hidden layer size", default=128)
    model_group.add_argument('-os', '--output_size', type=int, help="Output embeddings vector size", default=64)
    model_group.add_argument('-dm', '--distance_metric', type=str, help="Distance metric to use ('L1' or 'L2')", choices=['L1', 'L2'], default='L2')

    # running parameters
    running_group = parser.add_argument_group('Running')
    running_group.add_argument('-rs', '--random_seed', type=int, help="Random seed", default=42)
    running_group.add_argument('-lr', '--learning_rate', type=float, help="Learning rate", default=0.0001)
    running_group.add_argument('-ep', '--epochs', type=int, help="Number of epochs", default=1000)
    running_group.add_argument('-bs', '--batch_size', type=int, help="Batch size", default=32)
    running_group.add_argument('-sd', '--shuffle_data', action='store_true', help="Shuffle data", default=True)
    running_group.add_argument('-nw', '--num_workers', type=int, help="Number of workers", default=2)
    running_group.add_argument('-nr', '--num_runs', type=int, help="Number of evaluation runs", default=100)
    running_group.add_argument('-fd', '--feat_dim', type=int, help='Feature dimension', default=512)
    running_group.add_argument('-ks', '--k_shot', type=int, help='Number of samples per class for support', default=1)
    running_group.add_argument('-mn', '--model_name', type=str, help='Model name', default='siamese')

    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument('-dp', '--dataset_path', type=str, help='Dataset path', default='data/')
    dataset_group.add_argument('-dn', '--dataset_name', type=str, help='Dataset name', default='OlivettiFaces')

    # checkpoint parameters
    checkpoint_group = parser.add_argument_group('Checkpoints')
    checkpoint_group.add_argument('-sm', '--save_model', action='store_true', help="If the model should be saved", default=True)
    checkpoint_group.add_argument('-cd', '--checkpoint_dir', type=str, help="Directory where checkpoints will be stored", default='checkpoints/')

    # device parameters
    device_group = parser.add_argument_group('Device')
    device_group.add_argument('-ug', '--use_gpu', action='store_true', help="If training should use GPU", default=False)
    device_group.add_argument('-gd', '--gpu_device', type=str, help="GPU device identifier to use (ex: cuda:0)", default='cuda:0')

    # logging parameters
    logging_group = parser.add_argument_group('Log')
    logging_group.add_argument('-sl', '--save_log', action='store_true', help="If the log should be saved", default=False)
    logging_group.add_argument('-lp', '--log_path', type=str, help='Log path', default='logs/')
    logging_group.add_argument('-lf', '--log_file', type=str, help='Log file name', default='log.txt')
    logging_group.add_argument('-vl', '--verbose_level', type=int, help='Verbose level', default=2)

    self.args = parser.parse_args()

  def loader(self):
    """
    Loads a YAML configuration file specified in the `self.args.config` attribute.

    Returns:
      dict: The contents of the YAML file as a dictionary.

    Raises:
      FileNotFoundError: If the file specified in `self.args.config` does not exist.
      yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(self.args.config, 'r') as file:
      loaded_file = yaml.safe_load(file)
    return loaded_file

  def merge(self, config):
    """
    Merges the current argument values into the provided configuration dictionary.

    Args:
      config (dict): The configuration dictionary to be updated.

    Returns:
      dict: The updated configuration dictionary with the current argument values.
    """
    args_dict = vars(self.args)
    for key, value in args_dict.items():
      if value is not None:
        config = self.update(config, key.split('--')[-1], value)
    return config

  def update(self, config, key, value):
    """
    Update the value of a specified key in a configuration dictionary.

    Args:
      config (dict): The configuration dictionary where the key-value pair needs to be updated.
      key (str): The key whose value needs to be updated.
      value: The new value to be assigned to the specified key.

    Returns:
      dict: The updated configuration dictionary.
    """
    for section in config:
      if key in config[section]:
        config[section][key] = value
    return config
