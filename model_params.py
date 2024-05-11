import torch
def count_num_param(model=None, params=None):
    """Args:
            model (nn.Module): network model.
            params: network model's parameters.

        Examples:
            >>> model_size = count_num_param(model)
        """

    if model is not None:
        return sum(p.numel() for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].numel()
            else:
                s += p.numel()
        return s

    raise ValueError("At least one argument 'model' or 'params' must be provided.")