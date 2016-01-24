import lasagne
import theano
import theano.tensor as T
from lasagne import layers
from lasagne.regularization import regularize_network_params, l2


def create_iter_funcs_train(l_out, lr, mntm, wd):
    X = T.tensor4('X')
    y = T.ivector('y')
    X_batch = T.tensor4('X_batch')
    y_batch = T.ivector('y_batch')

    y_hat = layers.get_output(l_out, X, deterministic=False)

    # softmax loss
    train_loss = T.mean(
        T.nnet.categorical_crossentropy(y_hat, y))

    # L2 regularization
    train_loss += wd * regularize_network_params(l_out, l2)

    train_acc = T.mean(
        T.eq(y_hat.argmax(axis=1), y))

    all_params = layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        train_loss, all_params, lr, mntm)

    train_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[train_loss, train_acc],
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return train_iter


def create_iter_funcs_valid(l_out, bs=None, N=50, mc_dropout=False):
    X = T.tensor4('X')
    y = T.ivector('y')
    X_batch = T.tensor4('X_batch')
    y_batch = T.ivector('y_batch')

    if not mc_dropout:
        y_hat = layers.get_output(l_out, X, deterministic=True)
    else:
        if bs is None:
            raise ValueError('a fixed batch size is required for mc dropout')
        X_repeat = T.extra_ops.repeat(X, N, axis=0)
        y_sample = layers.get_output(
            l_out, X_repeat, deterministic=False)

        sizes = [X_repeat.shape[0] / X.shape[0]] * bs
        y_sample_split = T.as_tensor_variable(
            T.split(y_sample, sizes, bs, axis=0))
        y_hat = T.mean(y_sample_split, axis=1)

    valid_loss = T.mean(
        T.nnet.categorical_crossentropy(y_hat, y))
    valid_acc = T.mean(
        T.eq(y_hat.argmax(axis=1), y))

    valid_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[valid_loss, valid_acc],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return valid_iter


def create_iter_funcs_bayes_valid(l_out, bs, N=50):
    X = T.tensor4('X')
    y = T.ivector('y')
    X_batch = T.tensor4('X_batch')
    y_batch = T.ivector('y_batch')

    X_repeat = T.extra_ops.repeat(X, N, axis=0)
    y_hat = layers.get_output(
        l_out, X_repeat, deterministic=False)

    sizes = [X_repeat.shape[0] / X.shape[0]] * bs
    y_hat_split = T.as_tensor_variable(T.split(y_hat, sizes, bs, axis=0))
    y_mean = T.mean(y_hat_split, axis=1)
    valid_loss = T.mean(
        T.nnet.categorical_crossentropy(y_mean, y))
    valid_acc = T.mean(
        T.eq(y_mean.argmax(axis=1), y))

    valid_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[valid_loss, valid_acc],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return valid_iter


def create_iter_funcs_test(l_out, bs, N=50):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')

    X_repeat = T.extra_ops.repeat(X, N, axis=0)
    y_sample = layers.get_output(
        l_out, X_repeat, deterministic=False)

    # the number of splits needs to be pre-defined
    sizes = [X_repeat.shape[0] / X.shape[0]] * bs
    y_sample_split = T.as_tensor_variable(
        T.split(y_sample, sizes, bs, axis=0))
    y_hat = T.mean(y_sample_split, axis=1)
    #y_var = T.var(y_sample_split, axis=1)

    test_iter = theano.function(
        inputs=[theano.Param(X_batch)],
        outputs=y_hat,
        #outputs=[y_hat, y_var],
        givens={
            X: X_batch,
        },
    )

    return test_iter
