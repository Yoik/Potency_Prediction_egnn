import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import RankingDataset
from src.model import EfficiencyPredictor


def train_final_mask_model(
    all_query_feats,
    all_query_labels,
    all_query_ids,
    shared_ref_feats_by_pose,
    shared_ref_tensors,
    device,
    *,
    input_dim,
    dropout_rate,
    batch_size,
    learning_rate,
    weight_decay,
    num_epochs,
    l1_lambda
):
    final_ds = RankingDataset(
        query_feats=all_query_feats,
        query_labels=all_query_labels,
        query_ids=all_query_ids,
        ref_feats_by_pose=shared_ref_feats_by_pose,
        ref_value=1.0
    )

    final_loader = DataLoader(final_ds, batch_size=batch_size, shuffle=True)

    final_model = EfficiencyPredictor(
        input_dim=input_dim,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    final_model.train()
    print("Final Retraining...")

    for epoch in tqdm(range(num_epochs), desc="Final Mask Learning"):
        with torch.no_grad():
            ref_means = []
            for ref_t in shared_ref_tensors:
                out_ref = final_model.forward_one(ref_t.to(device))
                ref_means.append(torch.mean(out_ref["pred"]))
            score_ref_mean = torch.stack(ref_means).mean()

        for batch in final_loader:
            q_traj = batch['query_feat'].to(device)
            rank_label = batch['rank_label'].to(device)

            optimizer.zero_grad()

            out_q = final_model.forward_one(q_traj)
            score_q = out_q['pred'].view(-1)

            score_ref_batch = score_ref_mean.expand_as(score_q)

            loss = (
                torch.nn.functional.margin_ranking_loss(
                    score_q, score_ref_batch, rank_label, margin=0.3
                )
                + l1_lambda * torch.mean(out_q['mask'])
            )

            loss.backward()
            optimizer.step()

        scheduler.step()

    atom_mask_vals = torch.sigmoid(
        final_model.atom_mask_logits
    ).detach().cpu()

    global_mask_vals = torch.sigmoid(
        final_model.global_mask_logits
    ).detach().cpu()

    full_atom_mask = atom_mask_vals.repeat(9)
    final_mask = torch.cat(
        [full_atom_mask, global_mask_vals]
    ).numpy()

    return final_model, final_mask
