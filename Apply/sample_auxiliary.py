
from NN.lstm import *
from Util.chars_auxiliary_one_hot import *


def get_prim_one_hot_auxiliary_one_hot(chars_auxi, c, num_characters, appea, aroma, palat, taste, overa):
    char_int, auxi_one_hot = chars_auxi.get_char_int_auxi_onehot(c, appea, aroma, palat, taste, overa)
    char_ont_hot = np.eye(num_characters)[char_int]
    return np.concatenate(([[char_ont_hot]], [[auxi_one_hot]]), axis=2)

def get_prim_one_hot(chars_auxi, c, num_characters, auxiliary):
    char_int, auxi_one_hot = chars_auxi.get_char_int_auxi(c, auxiliary)
    char_ont_hot = np.eye(num_characters)[char_int]
    return np.concatenate(([[char_ont_hot]], [[auxi_one_hot]]), axis=2)


def pick_top_n(preds, vocab_size, top_n):
    p = np.squeeze(preds)
    _p = p.copy()

    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]  # get index
    return c, _p
    # return c, np.sort(_p)[::-1][:top_n]


def sample_auxiliary_one_hot(checkpoint, n_samples, lstm_size, num_characters,
           num_auxi, appea, aroma, palat, taste, overa,
           chars_auxi, prime, top_n=5):
    samples = []
    probability = []

    lstm = LSTM(num_characters=num_characters,
                num_auxiliary=num_auxi,
                lstm_size=lstm_size,
                sampling=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(lstm.model.initial_state)
        for c in prime:
            x = get_prim_one_hot_auxiliary_one_hot(chars_auxi, c, num_characters, appea, aroma, palat, taste, overa)
            feed = {lstm.model.inputs: x,
                    lstm.model.keep_prob: 1.,
                    lstm.model.initial_state: new_state}
            preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                        feed_dict=feed)

        _char, _ = pick_top_n(preds, num_characters, top_n)
        samples.append(chars_auxi.int_to_char[_char])

        for i in range(n_samples):
            x = get_prim_one_hot_auxiliary_one_hot(chars_auxi, chars_auxi.int_to_char[_char], num_characters, appea, aroma, palat, taste, overa)
            feed = {lstm.model.inputs: x,
                    lstm.model.keep_prob: 1.,
                    lstm.model.initial_state: new_state}
            preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                        feed_dict=feed)

            _char, _p = pick_top_n(preds, num_characters, top_n)
            samples.append(chars_auxi.int_to_char[_char])
            probability.append(_p)
    return ''.join(samples), np.array(probability)


def sample_auxiliary(checkpoint, n_samples, lstm_size, num_characters,
                     auxiliary, chars_auxi, prime, top_n=5):
    samples = []
    probability = []

    lstm = LSTM(num_characters=num_characters,
                num_auxiliary=len(auxiliary),
                lstm_size=lstm_size,
                sampling=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(lstm.model.initial_state)
        for c in prime:
            x = get_prim_one_hot(chars_auxi, c, num_characters, auxiliary)
            feed = {lstm.model.inputs: x,
                    lstm.model.keep_prob: 1.,
                    lstm.model.initial_state: new_state}
            preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                        feed_dict=feed)

        _char, _ = pick_top_n(preds, num_characters, top_n)
        samples.append(chars_auxi.int_to_char[_char])

        for i in range(n_samples):

            if u''.join(samples[len(samples) - 7:]) == u'<#end#>':
                return ''.join(samples[:len(samples) - 7]), np.array(probability)

            x = get_prim_one_hot(chars_auxi, chars_auxi.int_to_char[_char], num_characters, auxiliary)
            feed = {lstm.model.inputs: x,
                    lstm.model.keep_prob: 1.,
                    lstm.model.initial_state: new_state}
            preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                        feed_dict=feed)

            _char, _p = pick_top_n(preds, num_characters, top_n)
            samples.append(chars_auxi.int_to_char[_char])
            probability.append(_p)

        # while chars_auxi.int_to_char[_char] != '.':
        #     x = get_prim_one_hot(chars_auxi, chars_auxi.int_to_char[_char], num_characters, auxiliary)
        #     feed = {lstm.model.inputs: x,
        #             lstm.model.keep_prob: 1.,
        #             lstm.model.initial_state: new_state}
        #     preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
        #                                 feed_dict=feed)
        #
        #     _char, _p = pick_top_n(preds, num_characters, top_n)
        #     samples.append(chars_auxi.int_to_char[_char])
        #     probability.append(_p)
    return ''.join(samples), np.array(probability)



