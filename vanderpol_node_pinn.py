from typing import List
import numpy
import torch
from torchdiffeq import odeint_adjoint as odeint

# Set random seed for reproducibility
torch.manual_seed(42)
numpy.random.seed(42)


class VanDerPolODEFunc(torch.nn.Module):
    """Neural ODE model for the Van der Pol oscillator dynamics."""

    def __init__(self, hidden_dim: int = 32, n_layers: int = 3):
        super().__init__()

        # Build the neural network
        layers = []
        layers.append(torch.nn.Linear(2, hidden_dim))
        layers.append(torch.nn.Tanh())

        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(hidden_dim, 2))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, t, state):
        return self.net(state)


class VanDerPolPINN:
    """Physics-Informed Neural Network for Van der Pol oscillator."""

    def __init__(self, mu: float = 1.0, hidden_dim: int = 32, n_layers: int = 3):
        self.mu = mu
        self.rhs = VanDerPolODEFunc(hidden_dim, n_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rhs.to(self.device)

    def physics_loss(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on Van der Pol equation.
        dx/dt = y, dy/dt = μ(1-x²)y - x
        """
        # Ensure gradients are enabled for states
        states = states.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        x = states[:, 0:1]
        y = states[:, 1:2]

        # Get neural ODE predictions
        dx_dt = self.rhs(t, states)[:, 0:1]
        dy_dt = self.rhs(t, states)[:, 1:2]

        # Physics constraints for Van der Pol oscillator
        # 1) dx/dt
        physics_loss_1 = torch.mean((dx_dt - y) ** 2)

        # 2) dy/dt
        expected_dy_dt = self.mu * (1 - x**2) * y - x
        physics_loss_2 = torch.mean((dy_dt - expected_dy_dt) ** 2)

        return physics_loss_1 + physics_loss_2

    def data_loss(
        self, t: torch.Tensor, initial_state: torch.Tensor, target_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute data fitting loss."""
        # Solve ODE from initial condition
        predicted_states = odeint(self.rhs, initial_state, t, rtol=1e-6, atol=1e-8)

        # MSE loss
        return torch.mean((predicted_states - target_states) ** 2)

    def train(
        self,
        t_physics: torch.Tensor,
        t_data: torch.Tensor,
        initial_state: torch.Tensor,
        target_states: torch.Tensor,
        n_epochs: int = 5000,
        lr: float = 1e-6,
        physics_weight: float = 1.0,
    ) -> List[float]:
        """
        Train the PINN model.

        Args:
            t_physics: Time points for physics loss sampling
            t_data: Time points for data observations
            initial_state: Initial condition [x0, y0]
            target_states: Target trajectory at t_data points
            n_epochs: Number of training epochs
            lr: Learning rate
            physics_weight: Weight for physics loss term
        """
        optimizer = torch.optim.Adam(self.rhs.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5
        )

        # Move data to device
        t_physics = t_physics.to(self.device)
        t_data = t_data.to(self.device)
        initial_state = initial_state.to(self.device)
        target_states = target_states.to(self.device)

        losses = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample unique random points for physics loss (fixed the main bug)
            n_physics_points = min(100, len(t_physics))
            # Use torch.randperm to ensure unique indices
            idx = torch.randperm(len(t_physics))[:n_physics_points]
            t_sample = t_physics[idx]
            t_sample, _ = torch.sort(t_sample)  # Sort to ensure monotonic sequence

            # Generate states at sampled time points using current model
            with torch.enable_grad():
                states_sample = odeint(
                    self.rhs, initial_state, t_sample, rtol=1e-6, atol=1e-8
                )

            # Compute losses
            physics_loss = self.physics_loss(t_sample, states_sample)
            data_loss = self.data_loss(t_data, initial_state, target_states)

            total_loss = data_loss + physics_weight * physics_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            torch.torch.nn.utils.clip_grad_norm_(self.rhs.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            losses.append(total_loss.item())

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                    f"Data Loss = {data_loss.item():.6f}, "
                    f"Physics Loss = {physics_loss.item():.6f}"
                )

        return losses

    def predict(self, initial_state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict trajectory from initial state."""
        self.rhs.eval()
        with torch.no_grad():
            trajectory = odeint(
                self.rhs,
                initial_state.to(self.device),
                t.to(self.device),
                rtol=1e-6,
                atol=1e-8,
            )
        return trajectory.cpu()


def generate_vanderpol_data(
    mu: float, initial_state: numpy.ndarray, t: numpy.ndarray
) -> numpy.ndarray:
    """Generate ground truth Van der Pol oscillator data using scipy."""
    from scipy.integrate import odeint as scipy_odeint

    def vanderpol_dynamics(state, t):
        x, y = state
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        return [dxdt, dydt]

    trajectory = scipy_odeint(vanderpol_dynamics, initial_state, t)
    return trajectory


# Main execution
if __name__ == "__main__":
    # Parameters
    mu = 1.0
    t_start, t_end = 0.0, 20.0
    n_points = 200

    # Initial condition
    x0, y0 = 2.0, 0.0
    initial_state_np = numpy.array([x0, y0])

    # Generate time points
    t_np = numpy.linspace(t_start, t_end, n_points)

    # Generate ground truth data
    true_trajectory = generate_vanderpol_data(mu, initial_state_np, t_np)

    # Convert to PyTorch tensors
    t_data = torch.tensor(t_np[:50], dtype=torch.float32)  # Use subset for training
    t_physics = torch.tensor(t_np, dtype=torch.float32)
    initial_state = torch.tensor(initial_state_np, dtype=torch.float32)
    target_states = torch.tensor(true_trajectory[:50], dtype=torch.float32)

    # Create and train PINN
    pinn = VanDerPolPINN(mu=mu, hidden_dim=64, n_layers=4)

    print("Training PINN...")
    losses = pinn.train(t_physics, t_data, initial_state, target_states, n_epochs=3000)

    # Predict full trajectory
    t_test = torch.tensor(t_np, dtype=torch.float32)
    predicted_trajectory = pinn.predict(initial_state, t_test).numpy()

    # Compute errors
    mse_x = numpy.mean((predicted_trajectory[:, 0] - true_trajectory[:, 0]) ** 2)
    mse_y = numpy.mean((predicted_trajectory[:, 1] - true_trajectory[:, 1]) ** 2)
    print(f"\nMean Squared Errors:")
    print(f"MSE for x(t): {mse_x:.6f}")
    print(f"MSE for y(t): {mse_y:.6f}")
