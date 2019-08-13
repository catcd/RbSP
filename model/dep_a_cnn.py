import math
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.contrib.rnn import LSTMCell

from utils import Timer, Log, pad_sequences


class DepACNN:
    def __init__(self, model_name, embeddings, embeddings_wordnet_superset, constants):
        self.model_name = model_name
        self.embeddings = embeddings
        self.embeddings_wordnet_superset = embeddings_wordnet_superset

        self.input_w2v_dim = constants.INPUT_W2V_DIM
        self.use_word_embedding = constants.USE_WORD_EMBEDDINGS
        self.fine_tune_word_embedding = constants.FINE_TUNE_WORD_EMBEDDINGS

        self.char_embedding = constants.CHAR_EMBEDDING
        self.nchars = constants.NCHARS
        self.input_char_dim = 50
        self.output_lstm_char_dims = [50]
        self.char_cnn_filters = {2: 16, 3: 32, 4: 32}
        self.char_cnn_hidden_layers = 2

        self.use_pos = constants.USE_POS
        self.npos = constants.NPOS
        self.pos_embedding_dim = constants.POS_DIM

        self.use_wordnet_superset = constants.USE_WORDNET_SUPERSET
        self.wordnet_superset_dim = 45

        self.use_sent_lstm = True if len(constants.OUTPUT_SENT_LSTM_DIMS) != 0 else False
        self.output_sent_lstm_variant = constants.OUTPUT_SENT_LSTM_VARIANT
        self.output_sent_lstm_dims = constants.OUTPUT_SENT_LSTM_DIMS

        self.use_position = constants.USE_POSITION
        self.position_embedding_dim = constants.POSITION_DIM

        self.use_compositional_embedding = constants.USE_COMPOSITIONAL_EMBEDDING

        self.augmented_variant = constants.AUGMENTED_VARIANT
        assert self.augmented_variant.lower() in {'0', 'cnn', 'attention', 'concat', 'sequential'}, 'invalid augmented_variant'
        self.augmented_attention_variant = constants.AUGMENTED_ATTENTION_VARIANT
        assert self.augmented_attention_variant.lower() in {'normal', 'heuristic', 'sequential'}, 'invalid augmented_attention_variant'
        self.augmented_dim = constants.AUGMENTED_DIM
        self.use_augmented_word = constants.USE_AUGMENTED_WORD
        self.use_augmented_pos = constants.USE_AUGMENTED_POS
        self.use_augmented_rel = constants.USE_AUGMENTED_REL

        self.use_relation = constants.USE_RELATION
        self.nrelations = constants.NRELATIONS
        self.relation_embedding_dim = constants.RELATION_EMBEDDING_DIM

        self.use_direction = constants.USE_DIRECTION
        self.ndirections = constants.NDIRECTIONS
        self.direction_embedding_dim = constants.DIRECTION_EMBEDDING_DIM

        self.use_dependency = self.use_relation or self.use_direction

        self.model_variant = constants.MODEL_VARIANT
        assert self.model_variant.lower() in {'normal', 'dep_unit'}, 'invalid model_variant'
        self.cnn_filters = constants.CNN_FILTERS
        self.cnn_hidden_layers = constants.CNN_HIDEEN_LAYERS

        self.hidden_layers = constants.HIDDEN_LAYERS

        self.num_of_class = len(constants.ALL_LABELS)
        self.all_labels = constants.ALL_LABELS

        self.trained_models = constants.TRAINED_MODELS

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = tf.placeholder(name='labels', shape=[None], dtype='int32')

        self.word_ids = tf.placeholder(name='word_ids', dtype=tf.int32, shape=[None, None])
        self.sent_lengths = tf.placeholder(name='sent_lengths', dtype=tf.int32, shape=[None])
        self.pos_ids = tf.placeholder(name='pos_ids', dtype=tf.int32, shape=[None, None])
        self.char_ids = tf.placeholder(name='char_ids', shape=[None, None, None], dtype='int32')
        self.word_lengths = tf.placeholder(name='word_lengths', shape=[None, None], dtype='int32')
        self.s_relation_ids = tf.placeholder(name='s_relation_ids', dtype=tf.int32, shape=[None, None])
        self.wordnet_superset_ids = tf.placeholder(name='wordnet_superset_ids', shape=[None, None], dtype='int32')

        self.indexes = tf.placeholder(name='indexes', dtype=tf.int32, shape=[None, None])
        self.child_indexes = tf.placeholder(name='child_indexes', dtype=tf.int32, shape=[None, None, None])
        self.num_of_childs = tf.placeholder(name='num_of_childs', dtype=tf.int32, shape=[None, None])
        self.positions = tf.placeholder(name='positions', dtype=tf.float32, shape=[None, None, 2])
        self.child_positions = tf.placeholder(name='child_positions', dtype=tf.float32, shape=[None, None, None])

        self.relation_ids = tf.placeholder(name='relation_ids', dtype=tf.int32, shape=[None, None])
        self.direction_ids = tf.placeholder(name='direction_ids', dtype=tf.int32, shape=[None, None])

        self.dropout_augmented = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_augmented')
        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_embedding')
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.dropout_hidden_layer = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_hidden_layer')

        self.is_training = tf.placeholder(tf.bool, name='phase')

    @staticmethod
    def _deep_2d_cnn(cnn_input, embedding_dim, config, num_of_hidden_layers, dropout, stride_x=1, max_pooling=True):
        cnn_input = tf.expand_dims(cnn_input, -1)

        with tf.variable_scope('cnn_first_layer'):
            cnn_outputs = []
            for k in config:
                with tf.variable_scope('cnn-{}'.format(k)):
                    filters = config[k]
                    height = k * stride_x - 1 if stride_x == 2 else k

                    pad_top = stride_x * math.floor((k - 1) / 2)
                    pad_bottom = stride_x * math.ceil((k - 1) / 2)
                    temp_input = tf.pad(cnn_input, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])

                    cnn_op = tf.layers.conv2d(
                        temp_input, filters=filters,
                        kernel_size=(height, embedding_dim),
                        strides=(stride_x, 1),
                        padding='valid', name='cnn-{}'.format(k),
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=tf.nn.relu,
                    )  # batch, seq, 1, filter

                    cnn_outputs.append(cnn_op)
            cnn_output = tf.concat(cnn_outputs, axis=-1)
            cnn_output = tf.nn.dropout(cnn_output, dropout)

        for i in range(num_of_hidden_layers):
            with tf.variable_scope('cnn_hidden_layer-{}'.format(i + 1)):
                cnn_outputs = []
                for k in config:
                    with tf.variable_scope('cnn-{}'.format(k)):
                        filters = config[k]
                        height = k
                        pad_top = stride_x * math.floor((k - 1) / 2)
                        pad_bottom = stride_x * math.ceil((k - 1) / 2)
                        temp_input = tf.pad(cnn_output, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])

                        cnn_op = tf.layers.conv2d(
                            temp_input, filters=filters,
                            kernel_size=(height, 1),
                            padding='valid', name='cnn-{}'.format(k),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            activation=tf.nn.relu,
                        )

                        cnn_outputs.append(cnn_op)
                cnn_output = tf.concat(cnn_outputs, axis=-1)
                cnn_output = tf.nn.dropout(cnn_output, dropout)

        if max_pooling:
            final_cnn_output = tf.reduce_max(cnn_output, axis=[1, 2])
        else:
            final_cnn_output = tf.reduce_max(cnn_output, axis=2)

        return final_cnn_output

    @staticmethod
    def _multi_layer_bi_lstm(lstm_input, sequence_length, config, dropout, final_state_only=False):
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [LSTMCell(size) for size in config]
        )
        cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [LSTMCell(size) for size in config]
        )

        (output_fw, output_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw,
            lstm_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
        )

        if final_state_only:
            output_fw_f = final_state[0][-1][1]
            output_bw_f = final_state[1][-1][1]
            lstm_output = tf.concat([output_fw_f, output_bw_f], axis=-1)
        else:
            lstm_output = tf.concat([output_fw, output_bw], axis=-1)
        return tf.nn.dropout(lstm_output, dropout)

    @staticmethod
    def _self_embedding(input_ids, vocab_size, dimension, dropout):
        lookup_table = tf.get_variable(
            name='lut', dtype=tf.float32,
            shape=[vocab_size, dimension],
            initializer=tf.contrib.layers.xavier_initializer(),
            # regularizer=tf.contrib.layers.l2_regularizer(1e-4),
        )
        embeddings = tf.nn.embedding_lookup(lookup_table, input_ids, name='embedding')
        return tf.nn.dropout(embeddings, dropout)

    @staticmethod
    def _mlp_project(mlp_input, input_dim, config, num_of_class, dropout, with_time_step=False):
        if with_time_step:
            nsteps = tf.shape(mlp_input)[1]
            output = tf.reshape(mlp_input, [-1, input_dim])
        else:
            nsteps = 0
            output = mlp_input

        for i, v in enumerate(config, start=1):
            output = tf.layers.dense(
                inputs=output, units=v, name='hidden_{}'.format(i),
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                activation=tf.nn.tanh,
            )
            output = tf.nn.dropout(output, dropout)

        if num_of_class != 0:
            output = tf.layers.dense(
                inputs=output, units=num_of_class, name='final_dense',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )

        if with_time_step:
            if num_of_class != 0:
                return tf.reshape(output, [-1, nsteps, num_of_class])
            else:
                return tf.reshape(output, [-1, nsteps, config[-1]])
        else:
            return output

    @staticmethod
    def _mask_softmax(logits, seq_lens, max_seq_len, dim=-1):
        mask = tf.sequence_mask(seq_lens, max_seq_len, dtype=tf.float32)
        true_logits = tf.multiply(tf.exp(logits), mask)
        return tf.divide(true_logits, tf.reduce_sum(true_logits + 1e-9, dim, keep_dims=True))

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        relation_lut = tf.get_variable(
            name='relation_lut', dtype=tf.float32,
            shape=[self.nrelations, self.relation_embedding_dim],
            initializer=tf.contrib.layers.xavier_initializer(),
        )

        with tf.variable_scope('dependency_embedding'):
            if self.use_dependency:
                dependency_embedding_dim = 0

                if self.use_relation:
                    with tf.variable_scope('relation_embedding'):
                        embeddings = tf.nn.embedding_lookup(relation_lut, self.relation_ids, name='embedding')
                        relation_embeddings = tf.nn.dropout(embeddings, self.dropout_embedding)
                        dependency_embedding_dim += self.relation_embedding_dim
                else:
                    relation_embeddings = None

                if self.use_direction:
                    with tf.variable_scope('direction_embedding'):
                        direction_embeddings = self._self_embedding(
                            input_ids=self.direction_ids,
                            vocab_size=self.ndirections, dimension=self.direction_embedding_dim,
                            dropout=self.dropout_embedding,
                        )
                        dependency_embedding_dim += self.direction_embedding_dim
                        # LSTM on direction
                        # direction_embeddings = tf.nn.dropout(direction_embeddings, self.dropout_embedding)
                        #
                        # cell_fw = BasicLSTMCell(self.direction_embedding_dim)
                        # cell_bw = BasicLSTMCell(self.direction_embedding_dim)
                        #
                        # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        #     cell_fw, cell_bw,
                        #     direction_embeddings,
                        #     sequence_length=self.direction_seq_lens,
                        #     dtype=tf.float32,
                        # )
                        #
                        # direction_embeddings = tf.concat([output_fw, output_bw], axis=-1)
                else:
                    direction_embeddings = None

                if self.use_relation and self.use_direction:
                    dependency_embeddings = tf.concat([direction_embeddings, relation_embeddings], axis=-1)
                    self.dependency_embedding_dim = self.relation_embedding_dim + self.direction_embedding_dim
                    if self.use_compositional_embedding:
                        dependency_transform_dim = 100
                        dependency_embeddings = self._mlp_project(
                            mlp_input=dependency_embeddings, input_dim=self.dependency_embedding_dim,
                            config=[dependency_transform_dim], num_of_class=0,
                            dropout=self.dropout_hidden_layer, with_time_step=True,
                        )
                        self.dependency_embedding_dim = dependency_transform_dim
                elif self.use_relation:
                    self.dependency_embedding_dim = self.relation_embedding_dim
                    dependency_embeddings = relation_embeddings
                else:
                    self.dependency_embedding_dim = self.direction_embedding_dim
                    dependency_embeddings = direction_embeddings

                self.dependency_embeddings = dependency_embeddings
            else:
                self.dependency_embedding_dim = 1
                self.dependency_embeddings = tf.expand_dims(tf.zeros(tf.shape(self.direction_ids)), axis=-1)

        with tf.variable_scope('token_embedding'):
            token_embeddings = []
            token_embedding_dim = 0
            with tf.variable_scope('sent_rel_embedding'):
                sent_relation_embeddings = tf.nn.embedding_lookup(
                    relation_lut, self.s_relation_ids, name='embeddings'
                )
                sent_relation_embeddings = tf.nn.dropout(sent_relation_embeddings, self.dropout_embedding)

                # pad first relation for padding (use for lookup)
                self.padded_sent_rel_embeddings = tf.pad(
                    sent_relation_embeddings,
                    paddings=[[0, 0], [1, 0], [0, 0]]
                )

            with tf.variable_scope('word_info_embedding'):
                sent_word_info_embeddings = []
                sent_word_info_embedding_dim = 0
                if self.use_word_embedding:
                    with tf.variable_scope('sent_word_embedding'):
                        lut = tf.Variable(
                            self.embeddings, name='lut', dtype=tf.float32,
                            trainable=self.fine_tune_word_embedding
                        )
                        sent_word_embeddings = tf.nn.embedding_lookup(lut, self.word_ids, name='embeddings')
                        sent_word_embeddings = tf.nn.dropout(sent_word_embeddings, self.dropout_embedding)
                        sent_word_info_embeddings.append(sent_word_embeddings)
                        sent_word_info_embedding_dim += self.input_w2v_dim

                        # pad first token for padding (use for lookup)
                        self.padded_sent_word_embeddings = tf.pad(
                            sent_word_embeddings,
                            paddings=[[0, 0], [1, 0], [0, 0]]
                        )

                if self.char_embedding.lower() != '0':
                    with tf.variable_scope('sent_char_embedding'):
                        # batch, max_sent_length, seq, 50
                        char_embeddings = self._self_embedding(
                            input_ids=self.char_ids,
                            vocab_size=self.nchars, dimension=self.input_char_dim,
                            dropout=self.dropout_embedding
                        )

                        # put the time dimension on axis=1
                        s = tf.shape(char_embeddings)
                        char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.input_char_dim])
                        # batch*max_sent_length, seq, 50
                        word_lengths = tf.reshape(self.word_lengths, shape=[-1])

                        if self.char_embedding.lower() == 'lstm':
                            with tf.variable_scope('bi_lstm_char'):
                                # batch*max_sent_length, 50*2
                                lstm_output_char = self._multi_layer_bi_lstm(
                                    lstm_input=char_embeddings, sequence_length=word_lengths,
                                    config=self.output_lstm_char_dims, dropout=self.dropout_cnn,
                                    final_state_only=True
                                )
                                # batch, max_sent_length, 50*2
                                lstm_output_char = tf.reshape(
                                    lstm_output_char,
                                    shape=[-1, s[1], 2 * self.output_lstm_char_dims[-1]]
                                )

                            sent_word_info_embeddings.append(lstm_output_char)
                            sent_word_info_embedding_dim += 2 * self.output_lstm_char_dims[-1]
                        elif self.char_embedding.lower() == 'cnn':
                            with tf.variable_scope('cnn_char'):
                                cnn_output_char = self._deep_2d_cnn(
                                    cnn_input=char_embeddings, embedding_dim=self.input_char_dim,
                                    config=self.char_cnn_filters, num_of_hidden_layers=self.char_cnn_hidden_layers,
                                    dropout=self.dropout_cnn
                                )
                                output_char_embeddings_dim = sum(self.char_cnn_filters.values())
                                cnn_output_char = tf.reshape(
                                    cnn_output_char,
                                    shape=[-1, s[1], output_char_embeddings_dim]
                                )

                            sent_word_info_embeddings.append(cnn_output_char)
                            sent_word_info_embedding_dim += output_char_embeddings_dim

                if self.use_pos:
                    with tf.variable_scope('sent_pos_embedding'):
                        sent_pos_embeddings = self._self_embedding(
                            input_ids=self.pos_ids, vocab_size=self.npos, dimension=self.pos_embedding_dim,
                            dropout=self.dropout_embedding,
                        )
                        sent_word_info_embeddings.append(sent_pos_embeddings)
                        sent_word_info_embedding_dim += self.pos_embedding_dim

                        # pad first POS for padding (use for lookup)
                        self.padded_sent_pos_embeddings = tf.pad(
                            sent_pos_embeddings,
                            paddings=[[0, 0], [1, 0], [0, 0]]
                        )

                if self.use_wordnet_superset:
                    with tf.variable_scope('sent_wordnet_superset_embedding'):
                        wordnet_superset_lut = tf.Variable(
                            self.embeddings_wordnet_superset, name='lut', dtype=tf.float32, trainable=False
                        )
                        wordnet_superset_embeddings = tf.nn.embedding_lookup(
                            wordnet_superset_lut, self.wordnet_superset_ids, name='embeddings'
                        )
                        wordnet_superset_embeddings = tf.nn.dropout(wordnet_superset_embeddings, self.dropout_embedding)
                        sent_word_info_embeddings.append(wordnet_superset_embeddings)
                        sent_word_info_embedding_dim += self.wordnet_superset_dim

                if len(sent_word_info_embeddings) > 1:
                    sent_word_info_embeddings = tf.concat(sent_word_info_embeddings, axis=-1)
                elif len(sent_word_info_embeddings) == 1:
                    sent_word_info_embeddings = sent_word_info_embeddings[0]
                else:
                    sent_word_info_embeddings = None

                if sent_word_info_embeddings is not None:
                    if self.use_sent_lstm:
                        with tf.variable_scope('sent_token_embedding_lstm'):
                            sent_word_lstm_info = self._multi_layer_bi_lstm(
                                lstm_input=sent_word_info_embeddings, sequence_length=self.sent_lengths,
                                config=self.output_sent_lstm_dims, dropout=self.dropout_embedding, final_state_only=False,
                            )
                            if self.output_sent_lstm_variant.lower == 'c':
                                sent_word_info_embeddings = tf.concat([sent_word_info_embeddings, sent_word_lstm_info], axis=-1)
                                sent_word_info_embedding_dim += 2*self.output_sent_lstm_dims[-1]
                            else:
                                sent_word_info_embeddings = sent_word_lstm_info
                                sent_word_info_embedding_dim = 2*self.output_sent_lstm_dims[-1]

                    self.padded_sent_token_embeddings = tf.pad(
                        sent_word_info_embeddings,
                        paddings=[[0, 0], [1, 0], [0, 0]]
                    )

                    word_info_embeddings = tf.gather(params=self.padded_sent_token_embeddings, indices=self.indexes, axis=1)
                    batch_size = tf.shape(self.indexes)[0]
                    one_hot_mask = tf.one_hot(
                        tf.range(batch_size), depth=batch_size,
                        dtype=tf.bool, on_value=True, off_value=False,
                    )
                    word_info_embeddings = tf.boolean_mask(word_info_embeddings, mask=one_hot_mask, axis=0)
                    token_embeddings.append(word_info_embeddings)
                    token_embedding_dim += sent_word_info_embedding_dim

            if self.use_position:
                with tf.variable_scope('position_embedding'):
                    position_embeddings = self._mlp_project(
                        mlp_input=self.positions, input_dim=2,
                        config=[self.position_embedding_dim], num_of_class=0,
                        dropout=self.dropout_embedding, with_time_step=True,
                    )
                    token_embeddings.append(position_embeddings)
                    token_embedding_dim += self.position_embedding_dim

            if self.augmented_variant.lower() != '0':
                with tf.variable_scope('augmented_embedding'):
                    with tf.variable_scope('augmented_info'):
                        batch_size = tf.shape(self.indexes)[0]
                        augmented_info = []
                        augmented_info_dim = 0
                        if self.use_augmented_word and self.use_word_embedding:
                            with tf.variable_scope('augmented_word_embeddings'):
                                augmented_word_embeddings = tf.gather(params=self.padded_sent_word_embeddings, indices=self.child_indexes, axis=1)
                                one_hot_mask = tf.one_hot(
                                    tf.range(batch_size), depth=batch_size,
                                    dtype=tf.bool, on_value=True, off_value=False
                                )
                                augmented_word_embeddings = tf.boolean_mask(
                                    augmented_word_embeddings, mask=one_hot_mask, axis=0
                                )
                                augmented_info.append(augmented_word_embeddings)
                                augmented_info_dim += self.input_w2v_dim

                        if self.use_augmented_pos and self.use_pos:
                            with tf.variable_scope('augmented_pos_embeddings'):
                                augmented_pos_embeddings = tf.gather(params=self.padded_sent_pos_embeddings, indices=self.child_indexes, axis=1)
                                one_hot_mask = tf.one_hot(
                                    tf.range(batch_size), depth=batch_size,
                                    dtype=tf.bool, on_value=True, off_value=False
                                )
                                augmented_pos_embeddings = tf.boolean_mask(
                                    augmented_pos_embeddings, mask=one_hot_mask, axis=0
                                )
                                augmented_info.append(augmented_pos_embeddings)
                                augmented_info_dim += self.pos_embedding_dim

                        if self.use_augmented_rel:
                            with tf.variable_scope('augmented_relation_embeddings'):
                                augmented_rel_embeddings = tf.gather(params=self.padded_sent_rel_embeddings, indices=self.child_indexes, axis=1)
                                one_hot_mask = tf.one_hot(
                                    tf.range(batch_size), depth=batch_size,
                                    dtype=tf.bool, on_value=True, off_value=False
                                )
                                augmented_rel_embeddings = tf.boolean_mask(
                                    augmented_rel_embeddings, mask=one_hot_mask, axis=0
                                )
                                augmented_info.append(augmented_rel_embeddings)
                                augmented_info_dim += self.relation_embedding_dim

                        if len(augmented_info) > 1:
                            augmented_info = tf.concat(augmented_info, axis=-1)
                        elif len(augmented_info) == 1:
                            augmented_info = augmented_info[0]
                        else:
                            raise AttributeError('empty augmented info')

                    att_augmented_output = None
                    if self.augmented_variant.lower() in {'attention', 'concat', 'sequential'}:
                        with tf.variable_scope('augmented_attention'):
                            num_of_child = tf.shape(self.child_positions)[2]
                            max_seq_len = tf.shape(self.child_positions)[1]
                            reshaped_num_of_childs = tf.reshape(self.num_of_childs, shape=(-1,))
                            if self.augmented_attention_variant.lower() in {'normal', 'sequential'}:
                                with tf.variable_scope('self_attention'):
                                    with tf.variable_scope('attention_input'):
                                        # transform child position
                                        # (batch, seq, no) => (batch, seq, no, dim)
                                        child_position_dim = 32
                                        child_position = tf.expand_dims(self.child_positions, axis=-1)
                                        child_position = tf.reshape(child_position, shape=(-1, 1))
                                        child_position = tf.layers.dense(
                                            inputs=child_position, units=child_position_dim,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                            activation=tf.nn.tanh,
                                        )
                                        child_position = tf.reshape(
                                            child_position, shape=(-1, max_seq_len, num_of_child, child_position_dim),
                                        )
                                        child_position = tf.nn.dropout(child_position, self.dropout_embedding)

                                        # gather father word, POS for attention
                                        if self.use_word_embedding and self.use_pos:
                                            father_info = tf.concat(
                                                [self.padded_sent_word_embeddings, self.padded_sent_pos_embeddings],
                                                axis=-1,
                                            )
                                            father_info_dim = self.input_w2v_dim + self.pos_embedding_dim
                                        elif self.use_word_embedding:
                                            father_info = self.padded_sent_word_embeddings
                                            father_info_dim = self.input_w2v_dim
                                        elif self.use_pos:
                                            father_info = self.padded_sent_pos_embeddings
                                            father_info_dim = self.pos_embedding_dim
                                        else:
                                            father_info = tf.expand_dims(tf.zeros(tf.shape(self.word_ids)), axis=-1)
                                            father_info_dim = 1

                                        father_info = tf.gather(
                                            params=father_info,
                                            indices=self.indexes, axis=1,
                                        )
                                        batch_size = tf.shape(self.indexes)[0]
                                        one_hot_mask = tf.one_hot(
                                            tf.range(batch_size), depth=batch_size,
                                            dtype=tf.bool, on_value=True, off_value=False,
                                        )
                                        father_info = tf.boolean_mask(father_info, mask=one_hot_mask, axis=0)
                                        father_info = tf.expand_dims(father_info, axis=-2)
                                        father_info = tf.tile(father_info, [1, 1, num_of_child, 1])

                                        # concat attention input
                                        attention_input = tf.concat(
                                            [augmented_info, child_position, father_info],
                                            axis=-1
                                        )
                                        attention_input_dim = (
                                                augmented_info_dim
                                                + child_position_dim
                                                + father_info_dim
                                        )

                                    with tf.variable_scope('attention_weight'):
                                        # calculate attention weight
                                        attention_input = tf.reshape(
                                            attention_input,
                                            shape=[-1, attention_input_dim]
                                        )
                                        attention_hidden_layers = []
                                        attention_weights = self._mlp_project(
                                            mlp_input=attention_input, input_dim=attention_input_dim,
                                            config=attention_hidden_layers, num_of_class=1,
                                            dropout=self.dropout_hidden_layer, with_time_step=False,
                                        )
                                        attention_weights = tf.reshape(attention_weights, shape=(-1, num_of_child))
                                        attention_weights = self._mask_softmax(
                                            attention_weights, reshaped_num_of_childs, num_of_child
                                        )
                                        attention_weights = tf.reshape(
                                            attention_weights, shape=(-1, max_seq_len, num_of_child, 1)
                                        )

                                    # multiply element-wise with the input
                                    att_augmented_output = tf.multiply(attention_weights, augmented_info)

                            if self.augmented_attention_variant.lower() in {'heuristic', 'sequential'}:
                                with tf.variable_scope('heuristic_attention'):
                                    with tf.variable_scope('attention_weight'):
                                        # calculate attention weight
                                        alpha = tf.Variable(-0.03, trainable=False, dtype=tf.float32)
                                        attention_weights = tf.multiply(alpha, tf.square(self.child_positions))
                                        attention_weights = tf.reshape(attention_weights, shape=(-1, num_of_child))
                                        attention_weights = self._mask_softmax(
                                            attention_weights, reshaped_num_of_childs, num_of_child
                                        )
                                        attention_weights = tf.reshape(
                                            attention_weights, shape=(-1, max_seq_len, num_of_child, 1)
                                        )

                                        # multiply element-wise with the input
                                        att_augmented_output = tf.multiply(
                                            attention_weights,
                                            (augmented_info if self.augmented_attention_variant.lower() == 'heuristic' else att_augmented_output)
                                        )

                        if self.augmented_variant.lower() in {'attention', 'concat'}:
                            augmented_attention = tf.reduce_sum(att_augmented_output, axis=-2)

                            augmented_attention = self._mlp_project(
                                mlp_input=augmented_attention, input_dim=augmented_info_dim,
                                config=[self.augmented_dim], num_of_class=0,
                                dropout=self.dropout_embedding, with_time_step=True,
                            )
                            token_embeddings.append(augmented_attention)
                            token_embedding_dim += self.augmented_dim

                    if self.augmented_variant.lower() in {'cnn', 'concat', 'sequential'}:
                        with tf.variable_scope('augmented_cnn'):
                            if self.augmented_variant.lower() in {'cnn', 'concat'}:
                                augmented_cnn_input = augmented_info
                            else:
                                augmented_cnn_input = att_augmented_output

                            cnn_op = tf.layers.conv2d(
                                augmented_cnn_input, filters=self.augmented_dim,
                                kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                activation=tf.nn.relu,
                            )
                            augmented_cnn_features = tf.reduce_max(cnn_op, axis=2, keepdims=False)
                            augmented_cnn_features = tf.nn.dropout(augmented_cnn_features, self.dropout_augmented)

                        token_embeddings.append(augmented_cnn_features)
                        token_embedding_dim += self.augmented_dim

            if len(token_embeddings) >= 1:
                token_embeddings = tf.concat(token_embeddings, axis=-1)
                if self.use_compositional_embedding:
                    token_transform_dim = 300
                    self.token_embeddings = self._mlp_project(
                        mlp_input=token_embeddings, input_dim=token_embedding_dim,
                        config=[token_transform_dim], num_of_class=0,
                        dropout=self.dropout_hidden_layer, with_time_step=True,
                    )
                    self.token_embedding_dim = token_transform_dim
                else:
                    self.token_embeddings = token_embeddings
                    self.token_embedding_dim = token_embedding_dim
            elif len(token_embeddings) == 1:
                self.token_embedding_dim = token_embedding_dim
                self.token_embeddings = token_embeddings[0]
            else:
                self.token_embedding_dim = 1
                self.token_embeddings = tf.expand_dims(tf.zeros(tf.shape(self.indexes)), axis=-1)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        if self.model_variant == 'NORMAL':
            cnn_filter_width = max(self.token_embedding_dim, self.dependency_embedding_dim)
            cnn_step = 2

            padded_te = tf.pad(self.token_embeddings, [[0, 0], [0, 0], [0, cnn_filter_width - self.token_embedding_dim]])
            padded_de = tf.pad(self.dependency_embeddings, [[0, 0], [0, 1], [0, cnn_filter_width - self.dependency_embedding_dim]])

            max_seq_len = tf.shape(padded_te)[1]
            cnn_input = tf.stack([padded_te, padded_de], axis=-2)
            cnn_input = tf.reshape(cnn_input, shape=[-1, max_seq_len*2, cnn_filter_width])
        else:
            cnn_filter_width = 2*self.token_embedding_dim + self.dependency_embedding_dim
            cnn_step = 1

            cnn_input_component = []

            te1 = self.token_embeddings[:, :-1, :]
            cnn_input_component.append(te1)

            if self.use_dependency:
                cnn_input_component.append(self.dependency_embeddings)

            te2 = self.token_embeddings[:, 1:, :]
            cnn_input_component.append(te2)

            cnn_input = tf.concat(cnn_input_component, axis=-1)

        with tf.variable_scope('conv'):
            final_cnn_output = self._deep_2d_cnn(
                cnn_input=cnn_input, embedding_dim=cnn_filter_width,
                config=self.cnn_filters, num_of_hidden_layers=self.cnn_hidden_layers,
                dropout=self.dropout_cnn,
                stride_x=cnn_step
            )

        output_cnn_dim = sum(self.cnn_filters.values())

        with tf.variable_scope('logit'):
            self.logits = self._mlp_project(
                mlp_input=final_cnn_output, input_dim=output_cnn_dim,
                config=self.hidden_layers, num_of_class=self.num_of_class,
                dropout=self.dropout_hidden_layer, with_time_step=False,
            )
            self.predict = tf.nn.softmax(self.logits)

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(log_likelihood)
            self.loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))
            # self.train_op = optimizer.minimize(self.loss)

    def build(self):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._add_word_embeddings_op()

        self._add_logits_op()
        self._add_loss_op()

        self._add_train_op()
        # f = tf.summary.FileWriter('tensorboard')
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)
        timer.stop()

    def load_data(self, train, validation):
        """
        :param dataset.dataset.Dataset train:
        :param dataset.dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start('Loading data')

        self.dataset_train = train
        self.dataset_validation = validation

        print('Number of training examples:', len(self.dataset_train.labels))
        print('Number of validation examples:', len(self.dataset_validation.labels))
        timer.stop()

    def _accuracy(self, sess, feed_dict):
        feed_dict[self.dropout_augmented] = 1.0
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden_layer] = 1.0
        feed_dict[self.is_training] = False

        logits = sess.run(self.logits, feed_dict=feed_dict)
        accuracy = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, feed_dict[self.labels]):
            logit = np.argmax(logit)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        average = 'macro' if self.num_of_class > 2 else 'binary'
        return accuracy, f1_score(predict, exclude_label, average=average)

    def _loss(self, sess, feed_dict):
        feed_dict[self.dropout_augmented] = 1.0
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden_layer] = 1.0
        feed_dict[self.is_training] = False

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, data, batch_size):
        """

        :param dataset.dataset.Dataset data:
        :return:
        """
        start = 0

        while start < len(data.words):
            l_batch = data.labels[start:start + batch_size]

            w_batch = data.words[start:start + batch_size]
            c_batch = data.chars[start:start + batch_size]
            pos_batch = data.poses[start:start + batch_size]
            wns_batch = data.wordnet_supersets[start:start + batch_size]

            sr_batch = data.s_relations[start:start + batch_size]

            i_batch = data.indexes[start:start + batch_size]
            ic_batch = data.child_indexes[start:start + batch_size]
            p_batch = data.positions[start:start + batch_size]
            pc_batch = data.child_positions[start:start + batch_size]

            r_batch = data.relations[start:start + batch_size]
            d_batch = data.directions[start:start + batch_size]

            word_ids, sent_lengths = pad_sequences(w_batch, pad_tok=0, nlevels=1)
            char_ids, word_lengths = pad_sequences(c_batch, pad_tok=0, nlevels=2)
            pos_ids, _ = pad_sequences(pos_batch, pad_tok=0, nlevels=1)
            wn_superset_ids, _ = pad_sequences(wns_batch, pad_tok=0, nlevels=1)

            s_relation_ids, _ = pad_sequences(sr_batch, pad_tok=0, nlevels=1)

            indexes, _ = pad_sequences(i_batch, pad_tok=0, nlevels=1)
            child_indexes, num_of_childs = pad_sequences(ic_batch, pad_tok=0, nlevels=2)
            positions, _ = pad_sequences(p_batch, pad_tok=(0.0, 0.0), nlevels=1)
            child_positions, _ = pad_sequences(pc_batch, pad_tok=0.0, nlevels=2)
            relation_ids, _ = pad_sequences(r_batch, pad_tok=0, nlevels=1)
            direction_ids, _ = pad_sequences(d_batch, pad_tok=0, nlevels=1)

            labels = l_batch

            start += batch_size
            batch_data = {
                self.labels: labels,
                self.word_ids: word_ids,
                self.sent_lengths: sent_lengths,
                self.pos_ids: pos_ids,
                self.char_ids: char_ids,
                self.word_lengths: word_lengths,
                self.wordnet_superset_ids: wn_superset_ids,
                self.s_relation_ids: s_relation_ids,
                self.indexes: indexes,
                self.child_indexes: child_indexes,
                self.num_of_childs: num_of_childs,
                self.positions: positions,
                self.child_positions: child_positions,
                self.relation_ids: relation_ids,
                self.direction_ids: direction_ids,
            }

            yield batch_data

    def _train(self, epochs, batch_size, early_stopping=True, patience=10, verbose=True, stop_by='f1'):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=21)
        best_f1 = 0.0
        best_loss = float('inf')
        nepoch_noimp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                self.dataset_train.shuffle()

                for idx, batch_data in enumerate(self._next_batch(data=self.dataset_train, batch_size=batch_size)):
                    feed_dict = {
                        **batch_data,
                        self.dropout_embedding: 0.5,
                        self.dropout_cnn: 0.5,
                        self.dropout_augmented: 0.5,
                        self.dropout_hidden_layer: 0.5,
                        self.is_training: True,
                    }

                    _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        Log.log('Iter {}, Loss: {}'.format(idx, loss_train))

                Log.log('End epochs {}'.format(e + 1))
                # saver.save(sess, self.model_name + '_ep{}'.format(e + 1))

                if early_stopping:
                    if stop_by == 'f1':
                        # stop by F1
                        total_acc = []
                        total_f1 = []

                        for batch_data in self._next_batch(data=self.dataset_validation, batch_size=batch_size):
                            feed_dict = {
                                **batch_data
                            }

                            acc, f1 = self._accuracy(sess, feed_dict=feed_dict)
                            total_acc.extend(acc)
                            total_f1.append(f1)

                        f1 = np.mean(total_f1)
                        Log.log('F1 val: {}'.format(f1))
                        Log.log('Acc val: {}'.format(np.mean(total_acc)))
                        if f1 > best_f1:
                            saver.save(sess, self.model_name)
                            Log.log('Save the model at epoch {}'.format(e + 1))
                            best_f1 = f1
                            nepoch_noimp = 0
                        else:
                            nepoch_noimp += 1
                            Log.log('Number of epochs with no improvement: {}'.format(nepoch_noimp))
                            if nepoch_noimp >= patience:
                                print('Best F1: {}'.format(best_f1))
                                break
                    else:
                        # stop by loss
                        total_loss = []

                        for batch_data in self._next_batch(data=self.dataset_validation, batch_size=batch_size):
                            feed_dict = {
                                **batch_data
                            }

                            loss = self._loss(sess, feed_dict=feed_dict)
                            total_loss.append(loss)

                        val_loss = np.mean(total_loss)
                        Log.log('Val loss: {}'.format(val_loss))
                        if val_loss < best_loss:
                            saver.save(sess, self.model_name)
                            Log.log('Save the model at epoch {}'.format(e + 1))
                            best_loss = val_loss
                            nepoch_noimp = 0
                        else:
                            nepoch_noimp += 1
                            Log.log('Number of epochs with no improvement: {}'.format(nepoch_noimp))
                            if nepoch_noimp >= patience:
                                print('Best loss: {}'.format(best_loss))
                                break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def run_train(self, epochs, batch_size, early_stopping=True, patience=10):
        timer = Timer()
        timer.start('Training model...')
        self._train(epochs=epochs, batch_size=batch_size, early_stopping=early_stopping, patience=patience)
        timer.stop()

    def predict_on_test(self, test, predict_class=True):
        """

        :param predict_class:
        :param dataset.dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            # a = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, self.model_name)

            y_pred = []

            for batch_data in self._next_batch(data=test, batch_size=128):
                feed_dict = {
                    **batch_data,
                    self.dropout_augmented: 1.0,
                    self.dropout_embedding: 1.0,
                    self.dropout_cnn: 1.0,
                    self.dropout_hidden_layer: 1.0,
                    self.is_training: False,
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    if predict_class:
                        decode_sequence = np.argmax(logit)
                        y_pred.append(decode_sequence)
                    else:
                        y_pred.append(logit)

        return y_pred
