import numpy as np



def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights=None,
    axis=None,
):
    if weights is None:
        weights = np.ones(y.shape)

    # _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe



def prob_metric(pred, true):
    point_pred = np.mean(pred, axis=-1)
    quantile_pred = np.quantile(pred, axis=-1,q=(np.arange(9) + 1) / 10)
    quantile_pred = quantile_pred.transpose((1,2,3,0))
    # mean_mse = MSE(np.mean(point_pred, axis=1), np.mean(true, axis=1))
    mse = MSE(point_pred, true)
    crps = mqloss(true, quantile_pred, quantiles=(np.arange(9) + 1) / 10)

    # return mean_mse, mse
    return mse, crps
    # return mae, mse, rmse, mape, mspe