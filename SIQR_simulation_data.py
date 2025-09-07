import os
import argparse
import sys
import time

import torch
import torchsde


# needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SIQR_SDE(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        # parameters: (beta, alpha, gamma, delta)
        self.params = params
        self.N = population_size

    # For efficiency: implement drift and diffusion together
    def f_and_g(self, t, x):
        S, I, Q, R = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        beta, alpha, gamma, delta = self.params[:, 0], self.params[:, 1], self.params[:, 2], self.params[:, 3]

        p_inf = beta * S * I / self.N
        p_inf_sqrt = torch.sqrt(p_inf)
        p_rec = gamma * I
        p_qua = alpha * I
        p_qua_sqrt = torch.sqrt(p_qua)
        p_delta = delta * Q
        p_delta_sqrt = torch.sqrt(p_delta)
        p_rec_sqrt = torch.sqrt(p_rec)

        # Drift terms (f)
        f_S = -p_inf
        f_I = p_inf - p_qua - p_rec
        f_Q = p_qua - p_delta
        f_R = p_rec + p_delta

        f_term = torch.stack([f_S, f_I, f_Q, f_R], dim=-1)

        # Diffusion terms (g)
        g_S = torch.stack([-p_inf_sqrt, torch.zeros_like(p_inf_sqrt), torch.zeros_like(p_inf_sqrt), torch.zeros_like(p_inf_sqrt)], dim=-1)
        g_I = torch.stack([p_inf_sqrt, torch.sqrt(p_rec + p_qua), torch.zeros_like(p_inf_sqrt), torch.zeros_like(p_inf_sqrt)], dim=-1)
        g_Q = torch.stack([torch.zeros_like(p_qua_sqrt), p_qua_sqrt, -p_delta_sqrt, torch.zeros_like(p_inf_sqrt)], dim=-1)
        g_R = torch.stack([torch.zeros_like(p_inf_sqrt), p_rec_sqrt, p_delta_sqrt, torch.zeros_like(p_inf_sqrt)], dim=-1)

        g_term = torch.stack([g_S, g_I, g_Q, g_R], dim=-1)
        return f_term, g_term


def solve_siqr_sdes(
    num_samples,
    device,
    grid=10000,
    savegrad=False,
    save=False,
    filename="siqr_sde_data.pt",
    theta_loc=None,
    theta_covmat=None,
):
    ####### Change priors here ######
    if theta_loc is None or theta_covmat is None:
        theta_loc = torch.tensor([0.9, 0.1, 0.2, 0.2], device=device).log()  # beta, alpha, gamma, delta
        theta_covmat = torch.eye(4, device=device) * 0.5 ** 2

    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    params = prior.sample(torch.Size([num_samples])).exp()
    #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_infected = 5.0  # initial number of infected
    initial_quarantined = 0.0  # initial number of quarantined
    initial_recovered = 0.0  # initial number of recovered

    ## [susceptible, infected, quarantined, recovered]
    y0 = torch.tensor(
        num_samples * [[population_size - initial_infected, initial_infected, initial_quarantined, initial_recovered]],
        device=device,
    )  # starting point
    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SIQR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    start_time = time.time()
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    print("Simulation Time: %s seconds" % (end_time - start_time))

    save_dict = dict()
    idx_good = torch.where(ys[:, :, 1].mean(0) >= 1)[0]

    save_dict["prior_samples"] = params[idx_good].cpu()
    save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)
    save_dict["ys"] = ys[:, idx_good, 1].cpu()  # save all compartments: S, I, Q, R

    if savegrad:
        grads = (ys[2:, :, 1] - ys[:-2, :, 1]) / (2 * save_dict["dt"])
        save_dict["grads"] = grads[:, idx_good].cpu()

    # meta data
    save_dict["N"] = population_size
    save_dict["I0"] = initial_infected
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        torch.save(save_dict, f"data/{filename}")

    print("DONE.")
    return save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epidemic: solve SIQR equations")
    parser.add_argument("--num-samples", default=1000, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    if not os.path.exists("data"):
        os.makedirs("data")

    args = parser.parse_args()

    print("Generating initial training data...")
    solve_siqr_sdes(
        num_samples=args.num_samples,
        device=args.device,
        grid=10000,
        save=True,
        savegrad=False,
    )
    print("Generating initial test data...")
    test_data = []
    for i in range(3):
        dict_i = solve_siqr_sdes(
            num_samples=args.num_samples,
            device=args.device,
            grid=10000,
            save=False,
            savegrad=False,
        )
        test_data.append(dict_i)

    save_dict = {
        "prior_samples": torch.cat([d["prior_samples"] for d in test_data]),
        "ys": torch.cat([d["ys"] for d in test_data], dim=1),
        "dt": test_data[0]["dt"],
        "ts": test_data[0]["ts"],
        "N": test_data[0]["N"],
        "I0": test_data[0]["I0"],
    }
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]
    torch.save(save_dict, "data/siqr_sde_data_test.pt")
