
from sklearn.metrics.pairwise import cosine_similarity
from NN.lstm_per_user import *
# from Util.chars_auxiliary_one_hot import *


def get_word_by_cosine_similarity(word, word_vector, preds):
    sim = cosine_similarity(word_vector, preds)
    # print(sim)
    select_index = sim.argsort()[::-1][0]
    return word[select_index]


def pick_top_n(preds, vocab_size, top_n):
    p = np.squeeze(preds)

    _p = p.copy()

    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]  # get index
    return c, _p


def generate_output_vector(sess, lstm, n_samples, item_attention,
                           auxiliary, primes):
    generate_word = []
    probabiliry = []

    new_state = sess.run(lstm.model.initial_state)

    auxiliary = np.array(auxiliary).reshape(1, 1, -1)

    item_attention = np.array([item_attention]).reshape(1, 1)

    for prime in primes:

        prime_vector = np.array([prime]).reshape(1, 1)


        # print(prime_vector, prime_vector.shape)

        feed = {lstm.model.inputs: prime_vector,
                lstm.model.auxiliary: auxiliary,
                lstm.model.item_attention: item_attention,
                lstm.model.keep_prob: 1.,
                lstm.model.initial_state: new_state}
        preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                    feed_dict=feed)

        # print(preds)
        # print(np.sort(preds))
        # print(np.sort(preds)[0][-1])
        # print(np.argsort(preds)[0][-1])
        #
        # print(pick_top_n(preds, vector_length, 1))

    # print(prime_vector.shape, preds.shape)

    next_word = np.argsort(preds)[0][-1]
    generate_word.append(next_word)
    probabiliry.append(preds)

    for i in range(n_samples):

        # print(next_word)

        input_vector = np.array([next_word]).reshape(1, 1)

        feed = {lstm.model.inputs: input_vector,
                lstm.model.auxiliary: auxiliary,
                lstm.model.item_attention: item_attention,
                lstm.model.keep_prob: 1.,
                lstm.model.initial_state: new_state}
        preds, new_state = sess.run([lstm.model.preds, lstm.model.final_state],
                                    feed_dict=feed)

        # print(preds)
        # print(np.sort(preds))
        # print(np.sort(preds)[0][-1])
        # print(np.argsort(preds)[0][-1])

        next_word = np.argsort(preds)[0][-1]

        # print(next_word)

        generate_word.append(next_word)
        probabiliry.append(preds)

    return generate_word, probabiliry


def get_sample(checkpoint, n_samples, lstm_size, vector_length, num_item_attention,
               auxiliary, item_attention, prime):

    lstm = LSTM(vector_length=vector_length,
                num_auxiliary=len(auxiliary),
                num_item_attention=num_item_attention,
                lstm_size=lstm_size,
                sampling=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        generate_word_index, probabiliry = generate_output_vector(sess, lstm, n_samples,
                                                                  item_attention, auxiliary, prime)

    return generate_word_index, probabiliry



