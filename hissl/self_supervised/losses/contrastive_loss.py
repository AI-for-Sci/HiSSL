import tensorflow as tf


# =======================================================================
# A Simple Framework for Contrastive Learning of Visual Representations
# Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
# https://arxiv.org/abs/2002.05709
#
# Big Self-Supervised Models are Strong Semi-Supervised Learners
# Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
# https://arxiv.org/abs/2006.10029
# ========================================================================
def simclr_contrastive_loss(hidden,
                            hidden_norm=True,
                            temperature=0.1
                            ):
    """Compute loss for model.
    Args:
      hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.
    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    LARGE_NUM = 10e9
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = loss_a + loss_b #tf.reduce_mean(loss_a + loss_b)

    return loss #, logits_ab, labels


def byol_contrastive_loss(hidden):
    z1, z2 = tf.split(hidden, 2, 0)
    z1 = tf.math.l2_normalize(z1, axis=-1)  # (2*bs, 128)
    z2 = tf.math.l2_normalize(z2, axis=-1)  # (2*bs, 128)

    # similarities = tf.reduce_sum(tf.multiply(z1, z2), axis=1)
    # return 2 - 2 * tf.reduce_mean(similarities)
    """Byol's regression loss. This is a simple cosine similarity."""
    # normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return tf.reduce_sum((z1 - z2)**2, axis=-1)

