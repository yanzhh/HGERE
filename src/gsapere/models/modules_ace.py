# coding=utf-8
"""
Legacy ACE-experiment helper modules.

MFVI and GNN were only used by baseline/ACE models and are not part of the
active GSAP-ERE pipeline.  They are preserved here so the transformers/
sub-tree can be removed while keeping the code accessible.

Dependencies (already in utils/model_ere.py):
  BiafEncoder, CPDTrilinear, get_ent_mask1d, get_ent_mask2d, get_ent_mask3d
"""

import torch
import torch.nn as nn

from utils.model_ere import (
    BiafEncoder,
    CPDTrilinear,
    get_ent_mask1d,
    get_ent_mask2d,
    get_ent_mask3d,
)


class MFVI(nn.Module):
    def __init__(self, ent_dim, rel_dim, mem_dim, n_ent_labels, n_rel_labels, args):
        super().__init__()
        self.args = args
        self.ent_dim = ent_dim
        self.iter = args.n_iter
        self.re = n_ent_labels
        self.rr = n_rel_labels

        self.ter_scorer = CPDTrilinear(
            input_dim1=ent_dim,
            input_dim2=ent_dim,
            input_dim3=rel_dim,
            rank=mem_dim,
            output_dim=n_ent_labels ** 2 * n_rel_labels,
        )

        self.bin_scorer = BiafEncoder(
            input_dim1=rel_dim,
            input_dim2=rel_dim,
            output_dim=n_rel_labels ** 2,
            rank=mem_dim,
            factorize=True,
        )

    def _ter_potential(self, qs, qo, qr, jointscores):
        """
        qs, qo, qr: masked distributions
        qs: bs x ni x nei
        qo: bs x nj x nej
        qr: bs x ni x nj x nrij
        jointscores: bs x ni x nj x nei x nej x nrij
        return:
        Fs : bs x ns x nse
        Fo : bs x no x noe
        Fr : bs x ns x no x nr
        """
        qr = qr.unsqueeze(-2).unsqueeze(-2)
        qs = qs.unsqueeze(-2).unsqueeze(-1)
        qo = qo.unsqueeze(-3).unsqueeze(-2)
        Fso = (qr * jointscores).sum(-1)
        Fs = (qo * Fso).sum(axis=(-3, -1))
        Fo = (qs * Fso).sum(axis=(-4, -2))
        Fr = (qs.unsqueeze(-1) * qo.unsqueeze(-1) * jointscores).sum(axis=(-2, -3))
        return Fs, Fo, Fr

    def _sib_potential(self, qr, jointscores):
        """
        qr: bs x ns x no x nr
        jointscores: bs x ni x nj x nk x nrij x nrik; j,k are objects
        """
        bs, ne, _, _, nr, _ = jointscores.shape
        qrij = qr.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, 1, ne, 1, nr)
        qrik = qr.unsqueeze(-3).unsqueeze(-2).repeat(1, 1, ne, 1, nr, 1)
        Frij = (qrik * jointscores).sum(axis=(-3, -1))
        Frik = (qrij * jointscores).sum(axis=(-4, -2))
        return Frij + Frik

    def _cop_potential(self, qr, jointscores):
        """
        qr: bs x ns x no x nr
        jointscores: bs x ni x nj x nk x nrik x nrjk, k is the obj dim.
        """
        bs, ne, _, _, nr, _ = jointscores.shape
        qrik = qr.unsqueeze(-3).unsqueeze(-1).repeat(1, 1, ne, 1, 1, nr)
        qrjk = qr.unsqueeze(-4).unsqueeze(-2).repeat(1, ne, 1, 1, nr, 1)
        Frik = (qrik * jointscores).sum(axis=(-4, -1))
        Frjk = (qrjk * jointscores).sum(axis=(-5, -2))
        return Frik + Frjk

    def _gp_potential(self, qr, jointscores):
        """
        qr: bs x ns x no x nr
        jointscores: bs x ni x nj x nk x nrij x nrjk; j is obj for rij, j is sub for rjk
        """
        bs, ne, _, _, nr, _ = jointscores.shape
        qrij = qr.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, 1, ne, 1, nr)
        qrjk = qr.unsqueeze(-4).unsqueeze(-2).repeat(1, ne, 1, 1, nr, 1)
        Frij = (qrij * jointscores).sum(axis=(-3, -1))
        Frjk = (qrjk * jointscores).sum(axis=(-5, -2))
        return Frij + Frjk

    def mfvi_ternary(
        self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers
    ):
        bs, ne, _ = sub_reprs.shape
        batch_mask1d = get_ent_mask1d(ent_numbers)
        batch_mask2d = get_ent_mask2d(ent_numbers)
        jointscores = self.ter_scorer(
            sub_reprs.unsqueeze(-2).expand(rel_reprs.shape),
            obj_reprs.unsqueeze(-2).expand(rel_reprs.shape),
            rel_reprs,
        ).reshape(bs, ne, ne, self.re, self.re, self.rr)
        jointscores *= batch_mask2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        qsv = subscores.clone()
        qov = objscores.clone()
        qrv = relscores.clone()

        for _ in range(self.iter):
            qsv = qsv.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
            qov = qov.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
            qrv = qrv.masked_fill(~batch_mask2d.unsqueeze(-1), -1e4)
            qs = qsv.softmax(dim=-1)
            qo = qov.softmax(dim=-1)
            qr = qrv.softmax(dim=-1)
            Fs, Fo, Fr = self._ter_potential(qs, qo, qr, jointscores)
            qsv = qsv + Fs
            qov = qov + Fo
            qrv = qrv + Fr

        return qsv, qov, qrv

    def mfvi_hybrid(
        self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers
    ):
        bs, ne, _ = sub_reprs.shape
        batch_mask1d = get_ent_mask1d(ent_numbers)
        batch_mask2d = get_ent_mask2d(ent_numbers)
        batch_mask3d = get_ent_mask3d(ent_numbers)
        qsv = subscores.clone()
        qov = objscores.clone()
        qrv = relscores.clone()

        ter_scores = self.ter_scorer(
            sub_reprs.unsqueeze(-2).expand(rel_reprs.shape),
            obj_reprs.unsqueeze(-2).expand(rel_reprs.shape),
            rel_reprs,
        ).reshape(bs, ne, ne, self.re, self.re, self.rr)
        ter_scores *= batch_mask2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if "sib" in self.args.factor_type:
            sib_scores = self.bin_scorer(
                rel_reprs.unsqueeze(-2).repeat(1, 1, 1, ne, 1),
                rel_reprs.unsqueeze(-3).repeat(1, 1, ne, 1, 1),
            ).reshape(bs, ne, ne, ne, self.rr, self.rr)
            sib_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)
        if "cop" in self.args.factor_type:
            cop_scores = self.bin_scorer(
                rel_reprs.unsqueeze(-3).repeat(1, 1, ne, 1, 1),
                rel_reprs.unsqueeze(-4).repeat(1, ne, 1, 1, 1),
            ).reshape(bs, ne, ne, ne, self.rr, self.rr)
            cop_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)
        if "gp" in self.args.factor_type:
            gp_scores = self.bin_scorer(
                rel_reprs.unsqueeze(-2).repeat(1, 1, 1, ne, 1),
                rel_reprs.unsqueeze(-4).repeat(1, ne, 1, 1, 1),
            ).reshape(bs, ne, ne, ne, self.rr, self.rr)
            gp_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)

        for _ in range(self.iter):
            qsv = qsv.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
            qov = qov.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
            qrv = qrv.masked_fill(~batch_mask2d.unsqueeze(-1), -1e4)
            qs = qsv.softmax(dim=-1)
            qo = qov.softmax(dim=-1)
            qr = qrv.softmax(dim=-1)
            ter_fs, ter_fo, ter_fr = self._ter_potential(qs, qo, qr, ter_scores)
            frs = []
            if "sib" in self.args.factor_type:
                frs.append(self._sib_potential(qr, sib_scores))
            if "cop" in self.args.factor_type:
                frs.append(self._cop_potential(qr, cop_scores))
            if "gp" in self.args.factor_type:
                frs.append(self._gp_potential(qr, gp_scores))
            bin_fr = sum(frs)
            qsv = qsv + ter_fs
            qov = qov + ter_fo
            qrv = qrv + ter_fr + bin_fr

        return qsv, qov, qrv

    def forward(
        self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers
    ):
        if self.args.factor_type == "ternary":
            subscores, objscores, relscores = self.mfvi_ternary(
                sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers
            )
        elif self.args.factor_type in {
            "tersib", "tercop", "tergp", "tersibcop",
            "tersibgp", "tercopgp", "tersibcopgp",
        }:
            subscores, objscores, relscores = self.mfvi_hybrid(
                sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers
            )
        else:
            raise ValueError("We do not experiment on binary config")
        return subscores, objscores, relscores


class GNN(nn.Module):
    def __init__(self, ent_dim, rel_dim, dropout, args):
        super().__init__()
        self.args = args
        self.iter = args.n_iter
        mem_dim = args.mem_dim
        layernorm = args.layernorm
        self.dropout = nn.Dropout(dropout)

        self.proj_kv_s = nn.Linear(ent_dim, mem_dim)
        self.proj_kv_o = nn.Linear(ent_dim, mem_dim)
        self.proj_kv_rs = nn.Linear(rel_dim, mem_dim)
        self.proj_kv_ro = nn.Linear(rel_dim, mem_dim)
        self.attn_combine_s = nn.Sequential(
            nn.Linear(mem_dim + ent_dim, mem_dim), nn.GELU()
        )
        self.attn_combine_o = nn.Sequential(
            nn.Linear(mem_dim + ent_dim, mem_dim), nn.GELU()
        )
        self.sv = nn.Linear(mem_dim, 1, bias=False)
        self.ov = nn.Linear(mem_dim, 1, bias=False)
        self.fc_s = nn.Linear(mem_dim, ent_dim)
        self.fc_o = nn.Linear(mem_dim, ent_dim)

        self.proj_kv_r = nn.Linear(rel_dim, mem_dim)
        self.proj_kv_sr = nn.Linear(ent_dim, mem_dim)
        self.proj_kv_or = nn.Linear(ent_dim, mem_dim)
        self.attn_combine_r = nn.Sequential(
            nn.Linear(mem_dim + rel_dim, mem_dim), nn.GELU()
        )
        self.rv = nn.Linear(mem_dim, 1, bias=False)
        self.fc_r = nn.Linear(mem_dim, rel_dim)

        self.layernorm_s = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
        self.layernorm_o = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
        self.layernorm_r = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

    def update_rel(self, sub_reprs, obj_reprs, rel_reprs):
        res = rel_reprs
        bs, ne, _, _ = rel_reprs.shape
        hs = self.proj_kv_sr(sub_reprs).unsqueeze(-2).repeat(1, 1, ne, 1).unsqueeze(-2)
        ho = self.proj_kv_or(obj_reprs).unsqueeze(-3).repeat(1, ne, 1, 1).unsqueeze(-2)
        if self.args.attn_self:
            hr = self.proj_kv_r(rel_reprs).unsqueeze(-2)
            total_h = torch.cat([hr, hs, ho], dim=-2)
        else:
            total_h = torch.cat([hs, ho], dim=-2)
        ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        comb = torch.cat([ht, total_h], dim=-1)
        energy = self.attn_combine_r(comb)
        energy = self.rv(energy).squeeze(-1)
        attention = energy.softmax(dim=-1)
        output = torch.einsum("bijk,bijkd->bijd", attention, total_h.to(attention.dtype))
        output = self.dropout(self.fc_r(output)) + res
        output = self.layernorm_r(output)
        return output

    def update_sub(self, sub_reprs, rel_reprs, ent_numbers):
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = sub_reprs
        bs, ne, _, _ = rel_reprs.shape
        hs = self.proj_kv_s(res)
        hr = self.proj_kv_rs(rel_reprs)
        total_h = (
            torch.cat([hs.unsqueeze(-2), hr], dim=-2) if self.args.attn_self else hr
        )
        ht = sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()
        comb = torch.cat([ht, total_h], dim=-1)
        energy = self.attn_combine_s(comb)
        energy = self.sv(energy).squeeze(-1)
        attn_mask = torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        attn_mask = (attn_mask + ~batch_mask).bool()
        if self.args.attn_self:
            attn_self = torch.zeros((bs, ne, 1), device=self.args.device).bool()
            attn_mask = torch.cat((attn_self, attn_mask), axis=-1)
        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)
        output = torch.einsum("bij,bijd->bid", attention, total_h.to(attention.dtype))
        output = self.dropout(self.fc_s(output)) + res
        output = self.layernorm_s(output)
        return output

    def update_obj(self, obj_reprs, rel_reprs, ent_numbers):
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = obj_reprs
        bs, ne, _, _ = rel_reprs.shape
        ho = self.proj_kv_o(res)
        hr = self.proj_kv_ro(rel_reprs)
        total_h = (
            torch.cat([ho.unsqueeze(-3), hr], dim=-3) if self.args.attn_self else hr
        )
        ht = obj_reprs.unsqueeze(-3).repeat(1, total_h.shape[-3], 1, 1).contiguous()
        comb = torch.cat([ht, total_h], dim=-1)
        energy = self.attn_combine_o(comb)
        energy = self.ov(energy).squeeze(-1)
        attn_mask = torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        attn_mask = (attn_mask + ~batch_mask).bool()
        if self.args.attn_self:
            attn_self = torch.zeros((bs, 1, ne), device=self.args.device).bool()
            attn_mask = torch.cat((attn_self, attn_mask), axis=-2)
        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-2)
        output = torch.einsum("bij,bijd->bjd", attention, total_h.to(attention.dtype))
        output = self.dropout(self.fc_o(output)) + res
        output = self.layernorm_o(output)
        return output

    def aggregate(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
        sub_reprs_new = self.update_sub(sub_reprs, rel_reprs, ent_numbers)
        obj_reprs_new = self.update_obj(obj_reprs, rel_reprs, ent_numbers)
        rel_reprs_new = self.update_rel(sub_reprs, obj_reprs, rel_reprs)
        return sub_reprs_new, obj_reprs_new, rel_reprs_new

    def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
        mask1d = get_ent_mask1d(ent_numbers)
        mask2d = get_ent_mask2d(ent_numbers)
        for _ in range(self.iter):
            sub_reprs, obj_reprs, rel_reprs = self.aggregate(
                sub_reprs, obj_reprs, rel_reprs, ent_numbers
            )
            sub_reprs *= mask1d.unsqueeze(-1)
            obj_reprs *= mask1d.unsqueeze(-1)
            rel_reprs *= mask2d.unsqueeze(-1)
        return sub_reprs, obj_reprs, rel_reprs
