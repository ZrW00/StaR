from evaluator import *
import argparse, yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml file')
    args = parser.parse_args()

    if args.config is not None:
        # Load config file
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = argparse.Namespace(**config_dict)
        
        evaluator = load_evaluator(config)
        
        metrics = evaluator.evaluate()
    else:
        raise ValueError('Please provide a evalutation config file')
