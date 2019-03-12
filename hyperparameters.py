
class Hyperparameters(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.rnn_layers = args.num_rnn_layers
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.lr = args.lr
        self.bidirectional = args.bidirectional

        print()
        print("Loaded hyperparameters: {}".format(self.__dict__))
        print()
