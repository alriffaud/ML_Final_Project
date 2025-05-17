""" This module builds the SAKT model using Keras. """
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import MultiHeadAttention


def build_sakt(E, d_model=128, n_heads=4, d_ff=256, n_blocks=2, max_len=200,
               dropout=0.1):
    """ Builds the SAKT model.
    Args:
        E (int): Number of unique exercises.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward layer.
        n_blocks (int): Number of transformer blocks.
        max_len (int): Maximum length of the input sequences.
        dropout (float): Dropout rate.
    Returns:
        tf.keras.Model: The SAKT model.
    """
    inp_X = Input(shape=(max_len,), name='X', dtype='int32')
    inp_mask = Input(shape=(max_len,), name='mask')

    # Embedding layers
    token_emb = Embedding(input_dim=2*E, output_dim=d_model,
                          name='tok_emb')(inp_X)
    pos_emb = Embedding(input_dim=max_len, output_dim=d_model,
                        name='pos_emb')(
                    tf.range(start=0, limit=max_len, delta=1))
    x = token_emb + pos_emb

    for _ in range(n_blocks):
        # Multi-head attention (causal)
        attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads)(
                    query=x, value=x, key=x,
                    attention_mask=tf.linalg.band_part(
                        tf.ones((max_len, max_len)), -1, 0)  # causal mask
               )
        x = LayerNormalization()(x + Dropout(dropout)(attn))

        # Feed-forward
        ff = Dense(d_ff, activation='relu')(x)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + Dropout(dropout)(ff))

    # Pooling: take last valid position
    # We multiply by mask to zero-out pads, then sum and divide by sum(mask)
    mask_sum = tf.reduce_sum(inp_mask, axis=1, keepdims=True)
    eps = 1e-8
    weighted = tf.reduce_sum(x * tf.expand_dims(inp_mask, -1),
                             axis=1) / (mask_sum + eps)

    # Final prediction
    out = Dense(1, activation='sigmoid')(weighted)
    model = Model([inp_X, inp_mask], out)
    model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['AUC', 'accuracy'])
    return model
