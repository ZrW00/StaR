from dataPreprocessor import *
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml file')
    parser.add_argument('--mergeConfig', type=str, default="dataPreprocessorYamls/atlas/merge.yaml", help='Path to merge config yaml file')
    args = parser.parse_args()

    if args.config is not None:
        # Load config file
        # Single config
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = argparse.Namespace(**config_dict)
        
        processor = load_processor(config)
        trainProcessedData, testProcessedData, stateTrainProcessedData, noneStateTrainProcessedData = processor.preprocessMP()
        
        processor.saveData(trainProcessedData, testProcessedData, stateTrainProcessedData, noneStateTrainProcessedData)
    elif args.mergeConfig is not None:
        with open(args.mergeConfig, 'r') as f:
            mergeConfigDict = yaml.safe_load(f)

        mergeConfig = argparse.Namespace(**mergeConfigDict)
        
        merger = DataMerger(mergeConfig)
        mergeData = merger.mergeTrain()
        merger.save(mergeData)
    else:
        raise ValueError('Please provide a data config file or a merge config file')