import torch
import torch.nn as nn


class RiskAwareLoss(nn.Module):
    def __init__(self, alpha_over=1.0, alpha_under=1.5, lower_bound=-13, upper_bound=10):
        super(RiskAwareLoss, self).__init__()
        self.alpha_over = alpha_over  # Penalty multiplier for overestimation
        self.alpha_under = alpha_under  # Penalty multiplier for underestimation
        self.lower_bound = lower_bound  # Lower bound for the error range
        self.upper_bound = upper_bound  # Upper bound for the error range
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_rul, actual_rul):
        # Calculate the baseline loss (MSE in this case)
        base_loss = self.mse_loss(predicted_rul, actual_rul)

        # Calculate the prediction error
        error = predicted_rul - actual_rul

        # Apply penalties based on whether the error is outside the acceptable range
        overestimation_penalty = torch.maximum(error - self.upper_bound, torch.zeros_like(error))
        underestimation_penalty = torch.maximum(self.lower_bound - error, torch.zeros_like(error))

        # Compute the total penalty
        total_penalty = self.alpha_over * overestimation_penalty + self.alpha_under * underestimation_penalty

        # Combine the base loss with the penalties
        total_loss = base_loss + total_penalty.mean()

        return total_loss


class new_RiskAwareLoss(nn.Module):
    def __init__(self, alpha_normal=1.0, alpha_high=1.2, lower_bound=-13, upper_bound=10):
        super(new_RiskAwareLoss, self).__init__()
        self.alpha_normal = alpha_normal  # Penalty for errors within the acceptable range
        self.alpha_high = alpha_high  # Higher penalty for errors outside the range
        self.lower_bound = lower_bound  # Lower bound for the acceptable error range
        self.upper_bound = upper_bound  # Upper bound for the acceptable error range
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_rul, actual_rul):
        # Calculate the baseline loss (MSE in this case)
        #base_loss = self.mse_loss(predicted_rul, actual_rul)

        # Calculate the prediction error
        error = predicted_rul - actual_rul

        # Apply penalties based on whether the error is inside or outside the acceptable range
        penalty = torch.where((error > self.lower_bound) & (error < self.upper_bound),
                              self.alpha_normal * self.mse_loss(predicted_rul, actual_rul),  # Normal penalty within the range
                              self.alpha_high * self.mse_loss(predicted_rul, actual_rul))    # Higher penalty outside the range
        #penalty = penalty.mean()
        # Combine the base loss with the penalty
        total_loss = penalty.mean()

        return total_loss
