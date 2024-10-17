import argparse


class ArgumentsParser:

	def __init__(self):
		super(ArgumentsParser, self).__init__()
		self.parser = argparse.ArgumentParser(description="Sign Language Similarity Params")
		self.dataset_group = self.parser.add_argument_group('Dataset')
		self.model_group = self.parser.add_argument_group('Model')
		self.execution_group = self.parser.add_argument_group('Execution')
		self.checkpoint_group = self.parser.add_argument_group('Checkpoints')
		self.setup_arguments()


	def setup_arguments(self):
		self.dataset_group.add_argument('-dp', '--dataset_path', type=str, help='Dataset path', default='dataset/')
		self.dataset_group.add_argument('-dn', '--dataset_name', type=str, help='Dataset name', default='minds')

		self.model_group.add_argument('-is', '--input_size', type=int, help="Input size of the model", default=28)
		self.model_group.add_argument('-hs', '--hidden_size', type=int, help="Hidden layer size", default=128)
		self.model_group.add_argument('-os', '--output_size', type=int, help="Output embeddings vector size", default=64)
		self.model_group.add_argument('-mn', '--model_name', type=str, help='Model name', choices=['Resnet18', 'Resnet50', 'Siamese'], default='Siamese')

		self.execution_group.add_argument('-rs', '--random_seed', type=int, help="Random seed", default=42)
		self.execution_group.add_argument('-lr', '--learning_rate', type=float, help="Learning rate", default=0.0001)
		self.execution_group.add_argument('-ep', '--epochs', type=int, help="Number of epochs", default=100)
		self.execution_group.add_argument('-bs', '--batch_size', type=int, help="Batch size", default=32)
		# self.execution_group.add_argument('-sd', '--shuffle_data', action='store_true', help="Shuffle data", default=True)
		self.execution_group.add_argument('-nw', '--num_workers', type=int, help="Number of workers", default=2)
		self.execution_group.add_argument('-nr', '--num_runs', type=int, help="Number of evaluation runs", default=100)
		self.execution_group.add_argument('-fd', '--feat_dim', type=int, help='Feature dimension', default=128)
		self.execution_group.add_argument('-ks', '--k_shot', type=int, help='Number of samples per class for support', default=1)
		self.execution_group.add_argument('-ir', '--image_representation', type=str, help='Image Representation type', choices=['Skeleton-DML', 'Skeleton-Magnitude', 'SL-DML'], default='Skeleton-DML')

		self.checkpoint_group.add_argument('-sm', '--save_model', action='store_true', help="If the model should be saved", default=True)
		self.checkpoint_group.add_argument('-cd', '--checkpoint_dir', type=str, help="Directory where checkpoints will be stored", default='checkpoints/')

		# logging_group = parser.add_argument_group('Log')
		# logging_group.add_argument('-sl', '--save_log', action='store_true', help="If the log should be saved", default=False)
		# logging_group.add_argument('-lp', '--log_path', type=str, help='Log path', default='logs/')
		# logging_group.add_argument('-lf', '--log_file', type=str, help='Log file name', default='log.txt')
		# logging_group.add_argument('-vl', '--verbose_level', type=int, help='Verbose level', default=2)


	def parse_args(self):
		return self.parser.parse_args()
