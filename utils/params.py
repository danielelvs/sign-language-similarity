import argparse


class Params:

	def __init__(self):

		super(Params, self).__init__()

		parser = argparse.ArgumentParser(description='Sign Language Similarity Params')

		dataset_group = parser.add_argument_group('Dataset')
		dataset_group.add_argument('-dp', '--dataset_path', type=str, help='Dataset path', default='dataset/')
		dataset_group.add_argument('-dn', '--dataset_name', type=str, help='Dataset name', default='minds')

		network_group = parser.add_argument_group('Network')
		network_group.add_argument('-is', '--input_size', type=int, help="Input size of the model", default=28)
		network_group.add_argument('-hs', '--hidden_size', type=int, help="Hidden layer size", default=128)
		network_group.add_argument('-os', '--output_size', type=int, help="Output embeddings vector size", default=64)
		network_group.add_argument('-nt', '--network_name', type=str, help='Network name', choices=['resnet18', 'resnet50', 'siamese'], default='siamese')

		execution_group = parser.add_argument_group('Execution')
		execution_group.add_argument('-rs', '--random_seed', type=int, help="Random seed", default=42)
		execution_group.add_argument('-lr', '--learning_rate', type=float, help="Learning rate", default=0.0001)
		execution_group.add_argument('-ep', '--epochs', type=int, help="Number of epochs", default=1000)
		execution_group.add_argument('-bs', '--batch_size', type=int, help="Batch size", default=32)
		execution_group.add_argument('-sd', '--shuffle_data', action='store_true', help="Shuffle data", default=True)
		execution_group.add_argument('-nw', '--num_workers', type=int, help="Number of workers", default=2)
		execution_group.add_argument('-nr', '--num_runs', type=int, help="Number of evaluation runs", default=100)
		execution_group.add_argument('-fd', '--feat_dim', type=int, help='Feature dimension', default=512)
		execution_group.add_argument('-ks', '--k_shot', type=int, help='Number of samples per class for support', default=1)
		execution_group.add_argument('-rt', '--r_type', type=str, help='Representation to image type', choices=['Skeleton-DML', 'Skeleton-Magnitude', 'SL-DML'], default='Skeleton-DML')

		checkpoint_group = parser.add_argument_group('Checkpoints')
		checkpoint_group.add_argument('-sm', '--save_model', action='store_true', help="If the model should be saved", default=True)
		checkpoint_group.add_argument('-cd', '--checkpoint_dir', type=str, help="Directory where checkpoints will be stored", default='checkpoints/')

		# logging_group = parser.add_argument_group('Log')
		# logging_group.add_argument('-sl', '--save_log', action='store_true', help="If the log should be saved", default=False)
		# logging_group.add_argument('-lp', '--log_path', type=str, help='Log path', default='logs/')
		# logging_group.add_argument('-lf', '--log_file', type=str, help='Log file name', default='log.txt')
		# logging_group.add_argument('-vl', '--verbose_level', type=int, help='Verbose level', default=2)

		return parser.parse_args()
