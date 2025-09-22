import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # Done: Implement the forward function
        # Linear 1: z1 = x W1^T + b1
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        z1 = x @ W1.t() + b1

        # Activation f
        if self.f_function == 'relu':
            a1 = torch.clamp(z1, min=0.0)
        elif self.f_function == 'sigmoid':
            a1 = 1.0 / (1.0 + torch.exp(-z1))
        elif self.f_function == 'identity':
            a1 = z1
        else:
            raise ValueError(f"Unsupported f_function: {self.f_function}")

        # Linear 2: z2 = a1 W2^T + b2
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        z2 = a1 @ W2.t() + b2

        # Activation g
        if self.g_function == 'relu':
            y_hat = torch.clamp(z2, min=0.0)
        elif self.g_function == 'sigmoid':
            y_hat = 1.0 / (1.0 + torch.exp(-z2))
        elif self.g_function == 'identity':
            y_hat = z2
        else:
            raise ValueError(f"Unsupported g_function: {self.g_function}")

        # Cache for backward
        self.cache = dict(x=x, z1=z1, a1=a1, z2=z2, y_hat=y_hat)

        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # Done: Implement the backward function
        # Unpack cache and parameters
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        y_hat = self.cache['y_hat']

        W2 = self.parameters['W2']

        # Derivative through g activation
        if self.g_function == 'relu':
            dg_dz2 = (z2 > 0).to(z2.dtype)
        elif self.g_function == 'sigmoid':
            dg_dz2 = y_hat * (1.0 - y_hat)
        elif self.g_function == 'identity':
            dg_dz2 = torch.ones_like(z2)
        else:
            raise ValueError(f"Unsupported g_function: {self.g_function}")

        dJdz2 = dJdy_hat * dg_dz2

        # Gradients for second linear layer
        # dJ/dW2 = dJ/dz2^T @ a1
        self.grads['dJdW2'] = dJdz2.t() @ a1
        # dJ/db2 = sum over batch of dJ/dz2
        self.grads['dJdb2'] = dJdz2.sum(dim=0)

        # Backprop to a1
        dJda1 = dJdz2 @ W2

        # Derivative through f activation
        if self.f_function == 'relu':
            df_dz1 = (z1 > 0).to(z1.dtype)
        elif self.f_function == 'sigmoid':
            # a1 = sigmoid(z1)
            df_dz1 = a1 * (1.0 - a1)
        elif self.f_function == 'identity':
            df_dz1 = torch.ones_like(z1)
        else:
            raise ValueError(f"Unsupported f_function: {self.f_function}")

        dJdz1 = dJda1 * df_dz1

        # Gradients for first linear layer
        self.grads['dJdW1'] = dJdz1.t() @ x
        self.grads['dJdb1'] = dJdz1.sum(dim=0)


    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # Done: Implement the mse loss
    diff = y_hat - y
    loss = (diff * diff).mean()
    dJdy_hat = 2.0 * diff / y_hat.numel()

    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # Done: Implement the bce loss
    eps = 1e-7
    y_hat_clamped = torch.clamp(y_hat, eps, 1.0 - eps)
    # BCE mean over all elements
    loss = (-y * torch.log(y_hat_clamped) - (1.0 - y) * torch.log(1.0 - y_hat_clamped)).mean()
    # dJ/dy_hat with mean reduction
    dL_dy = -y / y_hat_clamped + (1.0 - y) / (1.0 - y_hat_clamped)
    dJdy_hat = dL_dy / y_hat.numel() * y_hat.numel() / y_hat.numel()  # keep shape, simplify below
    dJdy_hat = dL_dy / y_hat.numel()

    return loss, dJdy_hat

