import os
import argparse
from batch_gen import BatchGenerator
from model import Trainer

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mapping_path', type=str, required=True)
    parser.add_argument('--features_path_low', type=str, required=True)
    parser.add_argument('--features_path_high', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--vid_list_file', type=str, required=True)
    parser.add_argument('--vid_list_file_tst', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_rate', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--num_f_maps', type=int, default=64)

    return parser.parse_args()

def main():
    args = get_arguments()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Read mapping in the format: "<index> <label>"
    actions_dict = dict()
    with open(args.mapping_path, 'r') as f:
        for line in f:
            idx, action = line.strip().split(maxsplit=1)
            actions_dict[action] = int(idx)

    num_classes = len(actions_dict)

    # Load training data
    batch_gen = BatchGenerator(
        num_classes,
        actions_dict,
        args.gt_path,
        args.features_path_low,
        args.features_path_high,
        args.sample_rate
    )
    batch_gen.read_data(args.vid_list_file)

    # Load testing data
    batch_gen_tst = BatchGenerator(
        num_classes,
        actions_dict,
        args.gt_path,
        args.features_path_low,
        args.features_path_high,
        args.sample_rate
    )
    batch_gen_tst.read_data(args.vid_list_file_tst)

    # Initialize trainer
    trainer = Trainer(
        num_layers=args.num_layers,
        r1=2,
        r2=2,
        num_f_maps=args.num_f_maps,
        input_dim=None,
        num_classes=num_classes,
        channel_masking_rate=0.3
    )

    # Start training
    trainer.train(
        save_dir=args.model_dir,
        batch_gen=batch_gen,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        batch_gen_tst=batch_gen_tst
    )

if __name__ == "__main__":
    main()
