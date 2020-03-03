import tensorflow as tf

cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
cosine_similarity_dim1 = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=1)
cosine_similarity_dim2 = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=2)


def n_way_softmax(v_i, v_it, negatives, T=0.07):
    l_pos = tf.expand_dims(cosine_similarity_dim1(v_i, v_it), 1)
    l_pos /= T

    l_neg = cosine_similarity_dim2(tf.expand_dims(v_it, axis=1), tf.expand_dims(negatives, axis=0))
    l_neg /= T

    logits = tf.concat([l_pos, l_neg], axis=1)
    labels = tf.zeros(v_i.shape[0], dtype=tf.int32)
    h_loss = cross_entropy(y_true=labels, y_pred=logits)
    return h_loss


# @timeit
def nce_loss(f_vi, g_vit, negatives):
    assert f_vi.shape == g_vit.shape, "Shapes do not match" + str(f_vi.shape) + ' != ' + str(g_vit.shape)
    #  predicted input values of 0 and 1 are undefined (hence the clip by value)
    batch_size = f_vi.shape[0]
    return n_way_softmax(f_vi, g_vit, negatives) - tf.math.log(
        1 - tf.math.exp(-n_way_softmax(g_vit, negatives[:batch_size, :], negatives)))
