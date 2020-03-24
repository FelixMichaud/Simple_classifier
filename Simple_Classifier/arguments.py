import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--batch', default=0,
                            help="architecture of net_frame")
        parser.add_argument('--save_per_batchs', default=10,
                            help="the model, and pictures are save every this value")

        # Data related arguments
        parser.add_argument('--audRate', default=16000, type=int,
                            help='sound sampling rate')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        parser.add_argument('--mode', default='train', type=str,
                            help='weights regularizer')
        parser.add_argument('--path_inputs', default='', type=str,
                            help='path to save images')
        parser.add_argument('--path_targets', default='', type=str,
                            help='path to save images')
        parser.add_argument('--path_preds', default='', type=str,
                            help='path to save images')
        parser.add_argument('--starting_training_time', type=float,
                            help='strating time of the training')


        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        # optimization related arguments
        parser.add_argument('--lr_sound', default=1e-4, type=float, help='LR')
        self.parser = parser

#    def print_arguments(self, args):
#        file1 = open("MyFile.txt","a")
#        print("Input arguments:")
#        for key, val in vars(args).items():
#            file1.writelines([key, str(val), '\n']) 
#            print("{:16} {}".format(key, val))
#        file1.close()

#key=str        
#val=int
    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
#        self.print_arguments(args)
        return args
