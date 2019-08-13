import argparse

ALL_LABELS = [
    'Cause-Effect(e1,e2)', 'Component-Whole(e1,e2)', 'Content-Container(e1,e2)', 'Entity-Destination(e1,e2)',
    'Entity-Origin(e1,e2)', 'Instrument-Agency(e1,e2)', 'Member-Collection(e1,e2)', 'Message-Topic(e1,e2)',
    'Product-Producer(e1,e2)',
    'Other',
    'Cause-Effect(e2,e1)', 'Component-Whole(e2,e1)', 'Content-Container(e2,e1)', 'Entity-Destination(e2,e1)',
    'Entity-Origin(e2,e1)', 'Instrument-Agency(e2,e1)', 'Member-Collection(e2,e1)', 'Message-Topic(e2,e1)',
    'Product-Producer(e2,e1)'
]

parser = argparse.ArgumentParser(description='LSTM-CNN on Augmented SDP for Relation Extraction')

parser.add_argument('-i', help='Job identity', type=int, default=0)

parser.add_argument('-bs', help='Batch size default 128', type=int, default=128)
parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=10)

parser.add_argument('-per', help='Validation percentage', type=int, default=25)
parser.add_argument('-rp', help='Replacement', type=int, default=0)

parser.add_argument('-we', help='Use word embeddings', type=int, default=1)
parser.add_argument('-ftwe', help='Fine tune word embeddings', type=int, default=0)
parser.add_argument('-char', help='Use char embedding (0|cnn|lstm)', type=str, default='lstm')
parser.add_argument('-pos', help='POS tag embedding dimension', type=int, default=50)
parser.add_argument('-wns', help='Use WordNet superset', type=int, default=1)
parser.add_argument('-slstm', help='Number of output sentence LSTM dimension (prefix c|r)', type=str, default='0')
parser.add_argument('-pd', help='Position embedding dimension', type=int, default=0)

parser.add_argument('-com', help='Use compositional embedding', type=int, default=1)

parser.add_argument('-rel', help='Number of dependency relation embedding dimension', type=int, default=100)
parser.add_argument('-dir', help='Number of dependency direction embedding dimension', type=int, default=50)

parser.add_argument('-av', help='Augmented variant (0|CNN|ATTENTION|CONCAT|SEQUENTIAL)', type=str, default='SEQUENTIAL')
parser.add_argument('-aav', help='Augmented attention variant (NORMAL|HEURISTIC|SEQUENTIAL)', type=str, default='SEQUENTIAL')
parser.add_argument('-ad', help='Augmented dimension (CNN filter|Attention transform)', type=int, default=100)
parser.add_argument('-aw', help='Augmented by word', type=int, default=1)
parser.add_argument('-ap', help='Augmented by POS tag', type=int, default=1)
parser.add_argument('-ar', help='Augmented by dependency relation', type=int, default=1)

parser.add_argument('-mv', help='Model variant (NORMAL|DEP_UNIT)', type=str, default='NORMAL')
parser.add_argument('-cnn', help='CNN configurations', type=str, default='1:256,2:512,3:512')
parser.add_argument('-cnnh', help='Dept of hidden CNN', type=int, default=0)

parser.add_argument('-hd', help='Hidden layer configurations', type=str, default='0')

parser.add_argument('-skt', help='Skip training phase', type=int, default=0)

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i

BATCH_SIZE = opt.bs
EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p

PERCENTAGE = opt.per
REPLACEMENT = False if opt.rp == 0 else True

INPUT_W2V_DIM = 300

USE_WORD_EMBEDDINGS = False if opt.we == 0 else True
FINE_TUNE_WORD_EMBEDDINGS = False if opt.ftwe == 0 else True

CHAR_EMBEDDING = opt.char
NCHARS = 85

USE_POS = True if opt.pos != 0 else False
NPOS = 57
POS_DIM = opt.pos

USE_WORDNET_SUPERSET = True if opt.wns != 0 else False

OUTPUT_SENT_LSTM_VARIANT = opt.slstm[0]
OUTPUT_SENT_LSTM_DIMS = list(map(int, opt.slstm[1:].split(','))) if opt.slstm != '0' else []

USE_POSITION = True if opt.pd != 0 else False
POSITION_DIM = opt.pd

USE_COMPOSITIONAL_EMBEDDING = True if opt.com != 0 else False

USE_RELATION = False if opt.rel == 0 else True
NRELATIONS = 63
RELATION_EMBEDDING_DIM = opt.rel

USE_DIRECTION = False if opt.dir == 0 else True
NDIRECTIONS = 3
DIRECTION_EMBEDDING_DIM = opt.dir

AUGMENTED_VARIANT = opt.av
AUGMENTED_ATTENTION_VARIANT = opt.aav
AUGMENTED_DIM = opt.ad
USE_AUGMENTED_WORD = False if opt.aw == 0 else True
USE_AUGMENTED_POS = False if opt.ap == 0 else True
USE_AUGMENTED_REL = False if opt.ar == 0 else True

MODEL_VARIANT = opt.mv
CNN_FILTERS = {
    int(k): int(f) for k, f in [i.split(':') for i in opt.cnn.split(',')]
}
CNN_HIDEEN_LAYERS = opt.cnnh

HIDDEN_LAYERS = list(map(int, opt.hd.split(','))) if opt.hd != '0' else []

SKIP_TRAINING = opt.skt

DATA = 'data/sem2010/'
RAW_DATA = DATA + 'raw_data/'
PARSED_DATA = DATA + 'parsed_data/'
PICKLE_DATA = DATA + 'pickle/'

ALL_WORDS = PARSED_DATA + 'all_words.txt'
ALL_CHARS = PARSED_DATA + 'all_chars.txt'

ALL_POSES = PARSED_DATA + 'all_pos.txt'
ALL_RELATIONS = PARSED_DATA + 'all_relations.txt'

W2V_DATA = DATA + 'w2v_model/'
TRIMMED_W2V = W2V_DATA + 'sem2010_fasttext_wiki.npz'

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
