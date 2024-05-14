import torch
import torch.distributions as td
import torch.nn as nn

from src.models import GaussOutput


class KLSimMat(nn.Module):
    def __init__(self, sim_type: str) -> None:
        super().__init__()
        self.sim_type = sim_type

    def forward(
        self,
        pre: GaussOutput,
        ent: GaussOutput,
        con: GaussOutput,
    ):
        if self.sim_type == "pos":
            pos_mat = asymmetrical_kl_sim_mat(ent.mu, ent.std, pre.mu, pre.std)
            sim_mat = pos_mat

        elif self.sim_type == "pos+neg":
            pos_mat = asymmetrical_kl_sim_mat(ent.mu, ent.std, pre.mu, pre.std)
            neg_mat = asymmetrical_kl_sim_mat(con.mu, con.std, pre.mu, pre.std)
            sim_mat = torch.cat([pos_mat, neg_mat], dim=1)

        elif self.sim_type == "pos+rev":
            pos_mat = asymmetrical_kl_sim_mat(ent.mu, ent.std, pre.mu, pre.std)
            rev_mat = asymmetrical_kl_sim_mat(pre.mu, pre.std, ent.mu, ent.std)
            sim_mat = torch.cat([pos_mat, rev_mat], dim=1)

        elif self.sim_type == "pos+rev+neg":
            pos_mat = asymmetrical_kl_sim_mat(ent.mu, ent.std, pre.mu, pre.std)
            rev_mat = asymmetrical_kl_sim_mat(pre.mu, pre.std, ent.mu, ent.std)
            neg_mat = asymmetrical_kl_sim_mat(con.mu, con.std, pre.mu, pre.std)
            sim_mat = torch.cat([pos_mat, rev_mat, neg_mat], dim=1)

        else:
            raise ValueError(f"Invalid sim_type: {self.sim_type}")

        return sim_mat


# KLent_mat
def asymmetrical_kl_sim_mat(
    mu1: torch.FloatTensor,
    std1: torch.FloatTensor,
    mu2: torch.FloatTensor,
    std2: torch.FloatTensor,
) -> torch.Tensor:
    p1 = td.normal.Normal(mu1.unsqueeze(0), std1.unsqueeze(0))
    p2 = td.normal.Normal(mu2.unsqueeze(1), std2.unsqueeze(1))
    sim = 1 / (1 + td.kl.kl_divergence(p1, p2))

    return sim.mean(dim=-1)


# KLent
def asymmetrical_kl_sim(
    mu1: torch.FloatTensor,
    std1: torch.FloatTensor,
    mu2: torch.FloatTensor,
    std2: torch.FloatTensor,
) -> torch.Tensor:
    p1 = td.normal.Normal(mu1, std1)
    p2 = td.normal.Normal(mu2, std2)
    sim = 1 / (1 + td.kl.kl_divergence(p1, p2))

    return sim.mean(dim=-1)


# KLsim_mat
def symmetrical_kl_sim_mat(
    mu1: torch.FloatTensor,
    std1: torch.FloatTensor,
    mu2: torch.FloatTensor,
    std2: torch.FloatTensor,
) -> torch.FloatTensor:
    p1 = td.normal.Normal(mu1.unsqueeze(0), std1.unsqueeze(0))
    p2 = td.normal.Normal(mu2.unsqueeze(1), std2.unsqueeze(1))

    # CHECK: 二つの分布の平均KLダイバージェンスを計算するように変更
    # MEMO: JS Divergenceにしてもいいかもしれない
    kld1 = td.kl.kl_divergence(p1, p2)
    kld2 = td.kl.kl_divergence(p2, p1)
    mean_kld = (kld1 + kld2) / 2
    sim = 1 / (1 + mean_kld)

    return sim.mean(dim=-1)


def symmetrical_kl_sim(
    mu1: torch.FloatTensor,
    std1: torch.FloatTensor,
    mu2: torch.FloatTensor,
    std2: torch.FloatTensor,
) -> torch.FloatTensor:
    p1 = td.normal.Normal(mu1, std1)
    p2 = td.normal.Normal(mu2, std2)

    # CHECK: 二つの分布の平均KLダイバージェンスを計算するように変更
    # MEMO: JS Divergenceにしてもいいかもしれない
    kld1 = td.kl.kl_divergence(p1, p2)
    kld2 = td.kl.kl_divergence(p2, p1)
    mean_kld = (kld1 + kld2) / 2
    sim = 1 / (1 + mean_kld)

    return sim.mean(dim=-1)
