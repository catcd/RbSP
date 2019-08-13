import pickle

from data.sem2010 import constants
from model.dep_a_cnn import DepACNN
from utils import get_trimmed_w2v_vectors, load_wordnet_superset


def main():
    # train on full
    train = pickle.load(open(constants.PICKLE_DATA + 'sem2010.indexed.train.pickle', 'rb'))

    test = pickle.load(open(constants.PICKLE_DATA + 'sem2010.indexed.test.pickle', 'rb'))
    validation = train.clone(constants.PERCENTAGE, constants.REPLACEMENT, seed=constants.JOB_IDENTITY)

    # get pre trained embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
    _, wordnet_superset_embeddings = load_wordnet_superset()

    model = DepACNN(
        model_name=constants.MODEL_NAMES.format('sem2010', constants.JOB_IDENTITY),
        embeddings=embeddings,
        embeddings_wordnet_superset=wordnet_superset_embeddings,
        constants=constants,
    )

    # train, evaluate and interact
    model.build()
    model.load_data(train=train, validation=validation)
    if constants.SKIP_TRAINING == 0:
        model.run_train(constants.EPOCHS, constants.BATCH_SIZE, constants.EARLY_STOPPING, constants.PATIENCE)

    identities = test.identities
    y_pred = model.predict_on_test(test)
    of = open('data/output/answer-{}'.format(constants.JOB_IDENTITY), 'w')

    for i in range(len(y_pred)):
        of.write('{}\t{}\n'.format(identities[i][0], constants.ALL_LABELS[y_pred[i]]))

    of.close()


if __name__ == '__main__':
    main()
